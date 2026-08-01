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

"""Local, non-blocking browser controls for the DAgger state machine.

The HTTP handlers only enqueue the same abstract events used by keyboard and
pedal input. They never call a robot, teleoperator, or inference engine. Camera
preview encoding is performed by one background worker with a latest-only
queue so browser work cannot accumulate in the control loop.
"""

from __future__ import annotations

import errno
import json
import logging
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Event, Lock, Thread
from typing import Any
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CommandHandler = Callable[[], bool]


@dataclass(frozen=True)
class _CameraFrame:
    jpeg: bytes
    width: int
    height: int
    encoded_at_s: float
    sequence: int


class _LocalThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class DAggerWebUI:
    """Serve localhost DAgger controls and latest camera previews.

    ``command_handlers`` must contain the four phase-specific controls plus
    ``upload`` and ``stop`` callbacks. A callback returns ``True`` only when
    its request was accepted. Applying a phase transition remains the sole
    responsibility of the DAgger control loop.
    """

    _COMMANDS = {
        "pause",
        "resume",
        "start-correction",
        "finish-correction",
        "discard-correction",
        "upload",
        "stop",
    }
    _TRANSITION_COMMANDS = {
        "pause",
        "resume",
        "start-correction",
        "finish-correction",
        "discard-correction",
    }

    def __init__(
        self,
        *,
        port: int,
        auto_port: bool,
        preview_enabled: bool,
        preview_fps: float,
        jpeg_quality: int,
        command_handlers: Mapping[str, CommandHandler],
        task: str,
        target_episodes: int,
        discard_enabled: bool,
    ) -> None:
        missing = self._COMMANDS - set(command_handlers)
        if missing:
            raise ValueError(f"DAgger web UI is missing command handlers: {sorted(missing)}")

        self._requested_port = port
        self._auto_port = auto_port
        self._preview_enabled = preview_enabled
        self._preview_interval_s = 1.0 / preview_fps
        self._jpeg_quality = jpeg_quality
        self._command_handlers = dict(command_handlers)
        self._csrf_token = secrets.token_urlsafe(32)
        self._command_lock = Lock()

        self._state_lock = Lock()
        self._state: dict[str, Any] = {
            "phase": "autonomous",
            "recorded_episodes": 0,
            "target_episodes": target_episodes,
            "save_pending": False,
            "pending_command": None,
            "running": True,
            "task": task,
            "fault": None,
            "discard_enabled": discard_enabled,
        }
        self._camera_frames: dict[str, _CameraFrame] = {}
        self._camera_errors_reported: set[str] = set()

        self._encoder_condition = Condition()
        self._pending_images: dict[str, np.ndarray] | None = None
        self._next_preview_at_s = 0.0
        self._stop_event = Event()

        self._server: _LocalThreadingHTTPServer | None = None
        self._server_thread: Thread | None = None
        self._encoder_thread: Thread | None = None
        self._url: str | None = None

    @property
    def url(self) -> str:
        if self._url is None:
            raise RuntimeError("DAgger web UI has not started")
        return self._url

    def start(self) -> None:
        """Bind loopback and start HTTP/encoding worker threads."""
        if self._server is not None:
            raise RuntimeError("DAgger web UI is already started")

        handler_type = self._make_handler_type()
        try:
            try:
                server = _LocalThreadingHTTPServer(("127.0.0.1", self._requested_port), handler_type)
            except OSError as error:
                if not self._auto_port or self._requested_port == 0 or error.errno != errno.EADDRINUSE:
                    raise
                logger.warning(
                    "DAgger web UI port %d is unavailable; selecting a free localhost port",
                    self._requested_port,
                )
                server = _LocalThreadingHTTPServer(("127.0.0.1", 0), handler_type)

            self._server = server
            actual_port = int(server.server_address[1])
            self._url = f"http://127.0.0.1:{actual_port}"
            self._server_thread = Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="dagger-web-ui",
                daemon=True,
            )
            self._server_thread.start()

            if self._preview_enabled:
                self._encoder_thread = Thread(
                    target=self._encode_loop,
                    name="dagger-web-ui-encoder",
                    daemon=True,
                )
                self._encoder_thread.start()
        except BaseException:
            self._stop_event.set()
            with self._encoder_condition:
                self._encoder_condition.notify_all()
            if self._server is not None:
                if self._server_thread is not None and self._server_thread.is_alive():
                    self._server.shutdown()
                self._server.server_close()
            for thread in (self._server_thread, self._encoder_thread):
                if thread is not None and thread.is_alive():
                    thread.join(timeout=2.0)
            self._server = None
            self._server_thread = None
            self._encoder_thread = None
            self._url = None
            raise

        logger.info("DAgger operator UI: %s (localhost only)", self.url)

    def stop(self) -> None:
        """Stop accepting commands and terminate both background workers."""
        with self._state_lock:
            self._state["running"] = False
        self._stop_event.set()
        with self._encoder_condition:
            self._pending_images = None
            self._encoder_condition.notify_all()

        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()

        threads = (self._server_thread, self._encoder_thread)
        self._server_thread = None
        self._encoder_thread = None
        for thread in threads:
            if thread is None:
                continue
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError(f"DAgger web UI thread did not stop: {thread.name}")

    def publish_status(
        self,
        *,
        phase: str,
        recorded_episodes: int,
        save_pending: bool,
        fault: str | None = None,
    ) -> None:
        """Publish applied control state for browser polling."""
        with self._state_lock:
            self._state["phase"] = phase
            self._state["recorded_episodes"] = recorded_episodes
            self._state["save_pending"] = save_pending
            self._state["fault"] = fault

    def acknowledge_transition(self) -> None:
        """Clear a browser transition only after hardware side effects succeed."""
        with self._state_lock:
            if self._state["pending_command"] in self._TRANSITION_COMMANDS:
                self._state["pending_command"] = None

    def acknowledge_upload(self) -> None:
        with self._state_lock:
            if self._state["pending_command"] == "upload":
                self._state["pending_command"] = None

    def submit_observation(self, observation: Mapping[str, Any]) -> None:
        """Offer raw HWC uint8 camera arrays to a latest-only encoder queue."""
        if not self._preview_enabled or self._stop_event.is_set():
            return
        now_s = time.monotonic()
        with self._encoder_condition:
            if now_s < self._next_preview_at_s:
                return
            images = {
                name: value
                for name, value in observation.items()
                if isinstance(value, np.ndarray)
                and value.dtype == np.uint8
                and value.ndim == 3
                and value.shape[-1] == 3
            }
            if not images:
                return
            self._next_preview_at_s = now_s + self._preview_interval_s
            # Replace, rather than append. Slow browsers/encoders can never
            # create an unbounded backlog in the robot control loop.
            self._pending_images = images
            self._encoder_condition.notify()

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable state/camera snapshot."""
        now_s = time.monotonic()
        with self._state_lock:
            snapshot = dict(self._state)
            snapshot["cameras"] = {
                name: {
                    "width": frame.width,
                    "height": frame.height,
                    "sequence": frame.sequence,
                    "age_ms": max(0.0, (now_s - frame.encoded_at_s) * 1000.0),
                }
                for name, frame in sorted(self._camera_frames.items())
            }
        snapshot["preview_enabled"] = self._preview_enabled
        return snapshot

    def _request_command(self, command: str) -> bool:
        if command not in self._COMMANDS:
            return False

        # Serialize concurrent HTTP requests. Stop remains available as an
        # escape route, while all other commands wait for the control loop to
        # acknowledge the previous request.
        with self._command_lock:
            with self._state_lock:
                if command != "stop" and self._state["pending_command"] is not None:
                    return False
            accepted = self._command_handlers[command]()
            if accepted:
                with self._state_lock:
                    self._state["pending_command"] = command
            return accepted

    def _encode_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._encoder_condition:
                self._encoder_condition.wait_for(
                    lambda: self._pending_images is not None or self._stop_event.is_set(),
                    timeout=0.5,
                )
                if self._stop_event.is_set():
                    return
                images, self._pending_images = self._pending_images, None
            if images is None:
                continue

            for name, image in images.items():
                try:
                    # LeRobot camera observations use RGB; OpenCV's JPEG encoder
                    # expects BGR.
                    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                    )
                    if not ok:
                        raise RuntimeError("cv2.imencode returned false")
                    encoded_at_s = time.monotonic()
                    with self._state_lock:
                        previous = self._camera_frames.get(name)
                        self._camera_frames[name] = _CameraFrame(
                            jpeg=encoded.tobytes(),
                            width=int(image.shape[1]),
                            height=int(image.shape[0]),
                            encoded_at_s=encoded_at_s,
                            sequence=1 if previous is None else previous.sequence + 1,
                        )
                        self._camera_errors_reported.discard(name)
                except Exception as error:
                    with self._state_lock:
                        should_log = name not in self._camera_errors_reported
                        self._camera_errors_reported.add(name)
                    if should_log:
                        logger.warning("DAgger web UI cannot encode camera %s: %s", name, error)

    def _camera_frame(self, name: str) -> _CameraFrame | None:
        with self._state_lock:
            return self._camera_frames.get(name)

    def _make_handler_type(self) -> type[BaseHTTPRequestHandler]:
        ui = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "DAggerWebUI/1"

            def do_GET(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                if path == "/":
                    self._send_bytes(
                        HTTPStatus.OK,
                        ui._render_html(),
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/api/state":
                    self._send_json(HTTPStatus.OK, ui.snapshot())
                    return
                prefix = "/api/camera/"
                if path.startswith(prefix):
                    name = unquote(path.removeprefix(prefix))
                    frame = ui._camera_frame(name)
                    if frame is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "camera unavailable"})
                        return
                    self._send_bytes(
                        HTTPStatus.OK,
                        frame.jpeg,
                        "image/jpeg",
                        extra_headers={"X-Frame-Sequence": str(frame.sequence)},
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlsplit(self.path).path
                prefix = "/api/command/"
                if not path.startswith(prefix):
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return

                token = self.headers.get("X-DAgger-CSRF", "")
                if not secrets.compare_digest(token, ui._csrf_token):
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid CSRF token"})
                    return
                origin = self.headers.get("Origin")
                if origin is not None and origin not in {ui.url, ui.url.replace("127.0.0.1", "localhost")}:
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": "invalid origin"})
                    return

                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
                    return
                if content_length < 0 or content_length > 1024:
                    self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                    return
                if content_length:
                    self.rfile.read(content_length)

                command = unquote(path.removeprefix(prefix))
                try:
                    accepted = ui._request_command(command)
                except Exception:
                    logger.exception("DAgger web UI command handler failed: %s", command)
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "command failed"})
                    return
                if not accepted:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"accepted": False, "command": command, "error": "invalid transition"},
                    )
                    return
                self._send_json(HTTPStatus.ACCEPTED, {"accepted": True, "command": command})

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("DAgger web UI: " + format, *args)

            def _send_json(self, status: HTTPStatus, value: Any) -> None:
                self._send_bytes(
                    status,
                    json.dumps(value, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )

            def _send_bytes(
                self,
                status: HTTPStatus,
                payload: bytes,
                content_type: str,
                *,
                extra_headers: Mapping[str, str] | None = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' blob:; style-src 'unsafe-inline'; "
                    "script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; "
                    "form-action 'none'",
                )
                if extra_headers is not None:
                    for name, value in extra_headers.items():
                        self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    def _render_html(self) -> bytes:
        preview_interval_ms = max(100, int(self._preview_interval_s * 1000.0))
        html = _HTML.replace("__CSRF_TOKEN__", json.dumps(self._csrf_token)).replace(
            "__PREVIEW_INTERVAL_MS__", str(preview_interval_ms)
        )
        return html.encode("utf-8")


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LeRobot DAgger operator UI</title>
  <style>
    *{box-sizing:border-box} body{margin:0;background:#090a10;color:#e8eaf0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    main{max-width:1320px;margin:auto;padding:30px 22px 50px}.top,.panel{background:#151827;border:1px solid #292e45;border-radius:16px}
    .top{padding:22px 26px;display:flex;gap:22px;align-items:center;justify-content:space-between;flex-wrap:wrap}
    h1{font-size:21px;margin:0;color:#8cecff}.phase{font-size:25px;font-weight:800;letter-spacing:.08em;padding:10px 18px;border-radius:999px}
    .autonomous{background:#123b35;color:#74ffd8}.paused{background:#493c16;color:#ffe27a}.correcting{background:#501f32;color:#ff97bc}
    .meta{color:#9da4b8;line-height:1.7;margin-top:8px}.task{color:#d9deec}.panel{padding:22px;margin-top:20px}
    .controls{display:flex;gap:12px;flex-wrap:wrap}.button{border:1px solid #4e587a;background:#202640;color:#f7f8ff;border-radius:11px;padding:15px 20px;font:700 16px inherit;cursor:pointer}
    .button:hover:not(:disabled){background:#2b3457}.button.primary{border-color:#36d6bb;color:#7dffe7}.button.correction{border-color:#ff5d99;color:#ffa1c4}.button.discard,.button.stop{border-color:#e45b63;color:#ff9aa0}
    .button:disabled{opacity:.35;cursor:not-allowed}.notice{min-height:1.5em;margin-top:13px;color:#ffdd77}.error{color:#ff8189}
    .camera-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:15px;margin-top:14px}.camera{background:#090b13;border:1px solid #292e45;border-radius:12px;overflow:hidden}
    .camera header{display:flex;justify-content:space-between;padding:11px 13px;color:#aeb6cc}.camera img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#030408}.age{color:#69718a}
    @media(max-width:650px){main{padding:15px 10px}.phase{font-size:20px}.button{width:100%}}
  </style>
</head>
<body><main>
  <section class="top"><div><h1>LeRobot DAgger operator UI</h1><div id="task" class="meta task"></div><div id="episode" class="meta"></div></div><div id="phase" class="phase autonomous">AUTONOMOUS</div></section>
  <section class="panel"><div id="controls" class="controls"></div><div id="notice" class="notice"></div></section>
  <section class="panel"><strong>Camera previews</strong><div id="camera-note" class="meta">Waiting for frames…</div><div id="cameras" class="camera-grid"></div></section>
</main>
<script>
const CSRF=__CSRF_TOKEN__,PREVIEW_INTERVAL=__PREVIEW_INTERVAL_MS__;let latest=null,lastCameraRefresh=0;
const controls=document.getElementById('controls'),notice=document.getElementById('notice'),cards=new Map();
function button(label,command,kind='',confirmText=''){const b=document.createElement('button');b.className='button '+kind;b.textContent=label;b.disabled=Boolean(command!=='stop'&&latest&&latest.pending_command);b.onclick=async()=>{if(confirmText&&!confirm(confirmText))return;await send(command)};controls.appendChild(b)}
async function send(command){notice.textContent='Requesting '+command+'…';try{const r=await fetch('/api/command/'+command,{method:'POST',headers:{'X-DAgger-CSRF':CSRF}});const body=await r.json();if(!r.ok)throw new Error(body.error||('HTTP '+r.status));notice.textContent='Accepted; waiting for the control loop…'}catch(e){notice.textContent=e.message;notice.className='notice error'}}
function renderControls(){controls.replaceChildren();if(!latest||!latest.running)return;const p=latest.phase;if(p==='autonomous')button('Pause policy','pause','primary');if(p==='paused'){button('Start correction','start-correction','correction');button('Resume policy','resume','primary')}if(p==='correcting'){button('Finish correction & save','finish-correction','correction');if(latest.discard_enabled)button('Discard correction','discard-correction','discard','Discard this correction without saving it?')}button('Upload after shutdown','upload');button('Stop session','stop','stop','Stop DAgger and enter the normal safe shutdown flow?')}
function cameraCard(name){if(cards.has(name))return cards.get(name);const root=document.createElement('article');root.className='camera';const h=document.createElement('header'),label=document.createElement('span'),age=document.createElement('span'),img=document.createElement('img');label.textContent=name;age.className='age';img.alt=name+' preview';h.append(label,age);root.append(h,img);document.getElementById('cameras').append(root);const card={img,age,sequence:0};cards.set(name,card);return card}
function render(state){latest=state;const phase=document.getElementById('phase');phase.textContent=state.phase.toUpperCase();phase.className='phase '+state.phase;document.getElementById('task').textContent=state.task;document.getElementById('episode').textContent='Corrections: '+state.recorded_episodes+' / '+state.target_episodes+(state.save_pending?' · processing episode…':'');notice.className='notice'+(state.fault?' error':'');notice.textContent=state.fault||state.pending_command&&('Pending: '+state.pending_command)||'';renderControls();const names=Object.keys(state.cameras);document.getElementById('camera-note').textContent=state.preview_enabled?(names.length?names.length+' stream(s), latest-only preview':'Waiting for frames…'):'Preview disabled';const now=performance.now();for(const [name,info] of Object.entries(state.cameras)){const c=cameraCard(name);c.age.textContent=Math.round(info.age_ms)+' ms';if(info.sequence!==c.sequence&&now-lastCameraRefresh>=PREVIEW_INTERVAL){c.sequence=info.sequence;c.img.src='/api/camera/'+encodeURIComponent(name)+'?v='+info.sequence}}if(now-lastCameraRefresh>=PREVIEW_INTERVAL)lastCameraRefresh=now}
async function poll(){try{const r=await fetch('/api/state',{cache:'no-store'});if(!r.ok)throw new Error('state HTTP '+r.status);render(await r.json())}catch(e){notice.textContent='UI connection lost: '+e.message;notice.className='notice error'}}
poll();setInterval(poll,250);
</script></body></html>"""
