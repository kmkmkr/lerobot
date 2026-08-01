# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serialize episode persistence without blocking a hardware control loop.

``LeRobotDataset.save_episode`` mutates the dataset's current episode buffer
until persistence (including optional video encoding) has finished.  Recording
another frame at the same time is therefore unsafe.  :class:`AsyncEpisodeSaver`
provides an exclusive facade around that buffer:

* ``submit_save_episode`` and ``submit_discard_episode`` move buffer work to one
  background worker;
* ``add_frame`` rejects writes while an operation result awaits collection; and
* ``wait_for_pending_operation`` must succeed before the next recording starts.

The intended DAgger flow is to submit a completed correction as it enters the
paused phase, keep running the hardware hold/feedback loop, then collect the
result before accepting a transition into the next correction.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Literal, Protocol


class _WritableEpisodeDataset(Protocol):
    """Subset of the LeRobotDataset write API used by the saver."""

    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None: ...

    def clear_episode_buffer(self, delete_images: bool = True) -> None: ...

    def has_pending_frames(self) -> bool: ...


class AsyncEpisodeSaveError(RuntimeError):
    """Raised after a background save has failed and poisoned the writer."""


class AsyncEpisodeSaver:
    """Own exclusive access to one dataset's mutable episode buffer.

    All episode-buffer writes must go through :meth:`add_frame` while this
    helper is in use.  A caller that also needs to serialize other dataset
    operations (for example a Hub push) may pass its existing lock via
    ``dataset_lock`` and use the same lock around those operations.

    Args:
        dataset: A writable LeRobot dataset.
        dataset_lock: Optional lock shared with other dataset operations.
            The lock must implement the context-manager protocol.
        thread_name_prefix: Name prefix for the single save worker.
    """

    def __init__(
        self,
        dataset: _WritableEpisodeDataset,
        *,
        dataset_lock: Any | None = None,
        thread_name_prefix: str = "episode-save",
    ) -> None:
        self._dataset = dataset
        self._dataset_lock = dataset_lock if dataset_lock is not None else Lock()
        self._state_lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._pending_save: Future[None] | None = None
        self._pending_operation: Literal["save", "discard"] | None = None
        self._failure: BaseException | None = None
        self._closed = False

    @property
    def pending_operation(self) -> Literal["save", "discard"] | None:
        """Dataset operation whose result still needs to be collected."""
        with self._state_lock:
            return self._pending_operation

    @property
    def dataset_lock(self) -> Any:
        """Lock used to serialize calls that touch dataset state."""
        return self._dataset_lock

    @property
    def save_pending(self) -> bool:
        """Whether a save or discard result still needs to be collected.

        This remains true after the worker finishes.  The explicit collection
        requirement prevents a completed background exception from being
        silently ignored before the next correction starts.
        """
        with self._state_lock:
            return self._pending_save is not None

    @property
    def save_in_progress(self) -> bool:
        """Whether the background worker is actively saving or discarding."""
        with self._state_lock:
            return self._pending_save is not None and not self._pending_save.done()

    def _raise_if_unusable_locked(self) -> None:
        if self._failure is not None:
            raise AsyncEpisodeSaveError(
                "The previous episode operation failed; this dataset writer cannot safely record another episode."
            ) from self._failure
        if self._closed:
            raise RuntimeError("The asynchronous episode saver is closed.")

    def add_frame(self, frame: dict[str, Any]) -> None:
        """Add one frame when no dataset operation result is pending.

        A finished but uncollected operation still blocks writes. Call
        :meth:`wait_for_pending_operation` before starting the next recording.
        """
        with self._state_lock:
            self._raise_if_unusable_locked()
            if self._pending_save is not None:
                raise RuntimeError(
                    "Cannot add a frame while an episode operation result is pending; "
                    "call wait_for_pending_operation() (or wait_for_pending_save()) "
                    "before the next recording."
                )
            with self._dataset_lock:
                self._dataset.add_frame(frame)

    def submit_save_episode(self, *, parallel_encoding: bool = False) -> Future[None]:
        """Submit the current episode to the single save worker.

        ``parallel_encoding`` defaults to ``False`` so a multi-camera DAgger
        correction does not spawn one encoder process per camera while the
        hardware feedback loop is running.  The background save still keeps
        that loop responsive; cameras are encoded sequentially.

        Returns:
            The save future.  Callers should normally collect it through
            :meth:`wait_for_pending_save` rather than calling ``result``
            directly, because only the former releases frame admission.
        """
        with self._state_lock:
            self._raise_if_unusable_locked()
            if self._pending_save is not None:
                raise RuntimeError("An episode operation result is already pending collection.")
            with self._dataset_lock:
                if not self._dataset.has_pending_frames():
                    raise RuntimeError("Cannot save an episode with no pending frames.")

            future = self._executor.submit(self._save_episode, parallel_encoding)
            self._pending_save = future
            self._pending_operation = "save"
            return future

    def _save_episode(self, parallel_encoding: bool) -> None:
        with self._dataset_lock:
            self._dataset.save_episode(parallel_encoding=parallel_encoding)

    def submit_discard_episode(self) -> Future[None]:
        """Discard the current correction on the background worker.

        Clearing a video/image episode can wait for outstanding image writes
        and delete temporary files, so it must not run in the hardware loop.
        """
        with self._state_lock:
            self._raise_if_unusable_locked()
            if self._pending_save is not None:
                raise RuntimeError("An episode operation result is already pending collection.")
            future = self._executor.submit(self._discard_episode)
            self._pending_save = future
            self._pending_operation = "discard"
            return future

    def _discard_episode(self) -> None:
        with self._dataset_lock:
            self._dataset.clear_episode_buffer(delete_images=True)

    def wait_for_pending_operation(self) -> Literal["save", "discard"] | None:
        """Collect a pending save/discard and admit the next correction.

        Returns the completed operation name or ``None`` when nothing was
        pending. Any worker exception permanently poisons this helper because
        the mutable episode buffer may already have been partially changed.
        """
        with self._state_lock:
            if self._failure is not None:
                raise AsyncEpisodeSaveError(
                    "The previous episode operation failed; this dataset writer cannot safely continue."
                ) from self._failure
            pending = self._pending_save
            operation = self._pending_operation

        if pending is None:
            return None

        try:
            pending.result()
        except BaseException as exc:
            with self._state_lock:
                if self._pending_save is pending:
                    self._pending_save = None
                    self._pending_operation = None
                    self._failure = exc
            raise

        with self._state_lock:
            if self._pending_save is pending:
                self._pending_save = None
                self._pending_operation = None
        return operation

    def wait_for_pending_save(self) -> bool:
        """Compatibility wrapper that reports whether a save was collected.

        A completed discard returns ``False`` because it does not create data
        that needs uploading.
        """
        return self.wait_for_pending_operation() == "save"

    def shutdown(self) -> Literal["save", "discard"] | None:
        """Collect pending buffer work and stop the worker.

        The worker is always shut down, including when collection raises.  The
        first call propagates the original background exception; later calls
        are idempotent unless the saver is in the failed state.
        """
        with self._state_lock:
            if self._closed:
                if self._failure is not None:
                    raise AsyncEpisodeSaveError("The asynchronous episode saver failed.") from self._failure
                return None
            self._closed = True

        try:
            return self.wait_for_pending_operation()
        finally:
            self._executor.shutdown(wait=True)
