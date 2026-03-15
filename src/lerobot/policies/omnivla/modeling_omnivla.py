"""
OmniVLA Policy for LeRobot.

Wraps the OmniVLA Vision-Language-Action model for mobile robot navigation
into LeRobot's PreTrainedPolicy interface, enabling async inference support.

OmniVLA outputs trajectory waypoints as action chunks:
  (batch_size, chunk_size=8, action_dim=4)
where action_dim = (dx, dy, cos_heading, sin_heading).
"""

import os
import sys
import math
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.omnivla.configuration_omnivla import OmniVLAConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

# OmniVLA constants (matching prismatic/vla/constants.py)
OMNIVLA_NUM_ACTIONS_CHUNK = 8
OMNIVLA_ACTION_DIM = 4
OMNIVLA_POSE_DIM = 4
LEKIWI_ACTION_DIM = 3


def _add_omnivla_to_path(vla_path: str) -> str:
    """Add OmniVLA's parent directory to sys.path so prismatic modules can be imported."""
    # Try to find the OmniVLA repo relative to vla_path or in known locations
    candidates = [
        os.path.join(vla_path, ".."),  # vla_path is inside OmniVLA repo
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gitrepo", "OmniVLA"),
        os.path.join(os.getcwd(), "gitrepo", "OmniVLA"),
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        prismatic_path = os.path.join(candidate, "prismatic")
        if os.path.isdir(prismatic_path):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    raise FileNotFoundError(
        "Could not find OmniVLA prismatic module. "
        "Ensure the OmniVLA repository is available and vla_path is set correctly."
    )


class OmniVLAPolicy(PreTrainedPolicy):
    """LeRobot policy wrapper for OmniVLA mobile robot navigation model.

    This wraps the OmniVLA VLA model, action head, and pose projector
    into LeRobot's PreTrainedPolicy interface, supporting both synchronous
    and async inference.
    """

    config_class = OmniVLAConfig
    name = "omnivla"

    def __init__(self, config: OmniVLAConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # These will be initialized lazily on first use or via _load_omnivla_models
        self._vla = None
        self._action_head = None
        self._pose_projector = None
        self._action_tokenizer = None
        self._processor = None
        self._device_id = None
        self._num_patches = None
        self._models_loaded = False

        self.reset()

    def reset(self):
        """Clear action queue on environment reset."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def _ensure_models_loaded(self):
        """Lazily load OmniVLA models on first inference call."""
        if self._models_loaded:
            return

        omnivla_root = _add_omnivla_to_path(self.config.vla_path)

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from transformers import AutoConfig, AutoProcessor, AutoImageProcessor
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

        vla_path = self.config.vla_path
        if not os.path.isabs(vla_path):
            # Try relative to OmniVLA root
            candidate = os.path.join(omnivla_root, vla_path)
            if os.path.isdir(candidate):
                vla_path = candidate

        # Register OpenVLA model to HF Auto Classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

        device_id = torch.device(self.config.device if self.config.device else "cuda:0")

        # Load processor and VLA
        self._processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self._vla = AutoModelForVision2Seq.from_pretrained(
            vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device_id)

        self._vla.vision_backbone.set_num_images_in_input(self.config.num_images_in_input)
        self._vla.to(dtype=torch.bfloat16, device=device_id)

        # Load pose projector
        self._pose_projector = ProprioProjector(
            llm_dim=self._vla.llm_dim, proprio_dim=OMNIVLA_POSE_DIM
        )
        if self.config.resume_step is not None:
            state_dict = _load_checkpoint("pose_projector", vla_path, self.config.resume_step)
            self._pose_projector.load_state_dict(state_dict)
        self._pose_projector = self._pose_projector.to(device_id)

        # Load action head
        if self.config.use_l1_regression:
            self._action_head = L1RegressionActionHead_idcat(
                input_dim=self._vla.llm_dim,
                hidden_dim=self._vla.llm_dim,
                action_dim=OMNIVLA_ACTION_DIM,
            )
            if self.config.resume_step is not None:
                state_dict = _load_checkpoint("action_head", vla_path, self.config.resume_step)
                self._action_head.load_state_dict(state_dict)
            self._action_head = self._action_head.to(torch.bfloat16).to(device_id)

        # Compute number of vision patches
        self._num_patches = (
            self._vla.vision_backbone.get_num_patches()
            * self._vla.vision_backbone.get_num_images_in_input()
        )
        self._num_patches += 1  # for goal pose token

        # Create action tokenizer
        self._action_tokenizer = ActionTokenizer(self._processor.tokenizer)

        self._device_id = device_id
        self._models_loaded = True

    def get_optim_params(self) -> dict:
        """Return trainable parameters (not typically used for OmniVLA in LeRobot)."""
        self._ensure_models_loaded()
        params = list(self._action_head.parameters()) + list(self._pose_projector.parameters())
        return params

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions given observations.

        Expected batch keys:
            - "observation.images.current": (B, C, H, W) current camera image
            - "observation.images.goal": (B, C, H, W) goal image
            - "observation.state": (B, pose_dim) goal pose in local frame
                  [rel_y, -rel_x, cos(goal_heading - cur_heading), sin(goal_heading - cur_heading)]

        Optional batch keys:
            - "observation.language_instruction": str, language instruction

        Returns:
            - If config.convert_to_lekiwi_action is False (or return_waypoints=True), returns
              OmniVLA waypoints with shape (B, chunk_size, 4): (dx, dy, cos_heading, sin_heading).
            - Otherwise returns LeKiwi base velocity actions with shape (B, chunk_size, 3):
              (x.vel, y.vel, theta.vel), where y.vel is fixed to 0.0.
        """
        self._ensure_models_loaded()
        self.eval()

        # Build the OmniVLA-format batch from LeRobot batch
        omnivla_batch = self._prepare_omnivla_batch(batch)

        # Run forward pass
        actions, _ = self._run_forward_pass(omnivla_batch)

        waypoints = actions.float()
        if kwargs.get("return_waypoints", False) or not self.config.convert_to_lekiwi_action:
            return waypoints

        return self._convert_waypoints_to_lekiwi_actions(waypoints)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Select a single action from the action chunk queue.

        Manages an internal action queue: calls predict_action_chunk when
        the queue is empty, then pops one action per call.
        """
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """Compute loss for training (not the primary use case for OmniVLA in LeRobot).

        For training, use the original OmniVLA training scripts.
        This provides a basic L1 loss computation for compatibility.
        """
        self._ensure_models_loaded()

        # Predict actions
        predicted_actions = self.predict_action_chunk(batch, return_waypoints=True)

        # If ground truth actions are in the batch, compute L1 loss
        if ACTION in batch:
            gt_actions = batch[ACTION]
            loss = torch.nn.functional.l1_loss(predicted_actions, gt_actions)
            return loss, {"l1_loss": loss.item()}

        # If no ground truth, return zero loss
        loss = torch.tensor(0.0, device=predicted_actions.device, requires_grad=True)
        return loss, None

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _get_modality_id(self) -> torch.Tensor:
        """Determine modality ID based on config flags."""
        s = self.config.satellite
        l = self.config.lan_prompt
        p = self.config.pose_goal
        i = self.config.image_goal

        if s and not l and not p and not i:
            return torch.tensor([0], dtype=torch.float32)
        elif s and not l and p and not i:
            return torch.tensor([1], dtype=torch.float32)
        elif s and not l and not p and i:
            return torch.tensor([2], dtype=torch.float32)
        elif s and not l and p and i:
            return torch.tensor([3], dtype=torch.float32)
        elif not s and not l and p and not i:
            return torch.tensor([4], dtype=torch.float32)
        elif not s and not l and p and i:
            return torch.tensor([5], dtype=torch.float32)
        elif not s and not l and not p and i:
            return torch.tensor([6], dtype=torch.float32)
        elif not s and l and not p and not i:
            return torch.tensor([7], dtype=torch.float32)
        elif not s and l and p and not i:
            return torch.tensor([8], dtype=torch.float32)
        else:
            return torch.tensor([6], dtype=torch.float32)  # default: image_goal only

    def _prepare_omnivla_batch(self, batch: dict[str, Tensor]) -> dict:
        """Convert LeRobot-format batch to OmniVLA-format batch."""
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.vla.constants import ACTION_DIM

        # Extract images
        current_image = batch.get("observation.images.current")
        goal_image = batch.get("observation.images.goal")

        # Extract goal pose (state)
        goal_pose = batch.get(OBS_STATE)
        if goal_pose is None:
            goal_pose = torch.zeros(1, OMNIVLA_POSE_DIM)

        # Get language instruction
        lan_inst = self.config.language_instruction if self.config.lan_prompt else "xxxx"

        # Convert images to PIL for OmniVLA processor
        from PIL import Image
        import torchvision.transforms.functional as TF

        batch_size = 1
        if current_image is not None:
            if current_image.dim() == 4:
                batch_size = current_image.shape[0]
            current_image_pil = _tensor_to_pil(current_image[0] if current_image.dim() == 4 else current_image)
        else:
            current_image_pil = Image.new("RGB", (224, 224))

        if goal_image is not None:
            goal_image_pil = _tensor_to_pil(goal_image[0] if goal_image.dim() == 4 else goal_image)
        else:
            goal_image_pil = Image.new("RGB", (224, 224))

        if goal_pose.dim() == 2:
            goal_pose_np = goal_pose[0].cpu().numpy()
        else:
            goal_pose_np = goal_pose.cpu().numpy()

        # Create dummy actions for tokenization (required by OmniVLA's data pipeline)
        dummy_actions = np.random.rand(OMNIVLA_NUM_ACTIONS_CHUNK, OMNIVLA_ACTION_DIM)

        # Build data instance using OmniVLA's transform pipeline
        data_instance = self._transform_datatype(
            lan_inst, dummy_actions, goal_pose_np,
            current_image_pil, goal_image_pil,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self._action_tokenizer,
            base_tokenizer=self._processor.tokenizer,
            image_transform=self._processor.image_processor.apply_transform,
        )

        # Collate into batch
        omnivla_batch = self._collate([data_instance])
        return omnivla_batch

    def _transform_datatype(self, inst_obj, actions, goal_pose_cos_sin,
                            current_image_pil, goal_image_pil, prompt_builder,
                            action_tokenizer, base_tokenizer, image_transform):
        """Transform data into OmniVLA's expected format."""
        IGNORE_INDEX = -100

        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = ''.join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {"from": "human", "value": f"What action should the robot take to {inst_obj}?"},
                {"from": "gpt", "value": action_chunk_string},
            ]

        pb = prompt_builder("openvla")
        for turn in conversation:
            pb.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[:-(action_chunk_len + 1)] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_pil)
        pixel_values_goal = image_transform(goal_image_pil)

        return dict(
            pixel_values_current=pixel_values_current,
            pixel_values_goal=pixel_values_goal,
            input_ids=input_ids,
            labels=labels,
            actions=torch.as_tensor(actions, dtype=torch.float32),
            goal_pose=torch.as_tensor(goal_pose_cos_sin, dtype=torch.float32),
        )

    def _collate(self, instances):
        """Collate a list of data instances into a batch."""
        from torch.nn.utils.rnn import pad_sequence

        IGNORE_INDEX = -100
        pad_token_id = self._processor.tokenizer.pad_token_id
        model_max_length = self._processor.tokenizer.model_max_length

        input_ids = pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True, padding_value=pad_token_id
        )
        labels = pad_sequence(
            [inst["labels"] for inst in instances],
            batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, :model_max_length]
        labels = labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
            pixel_values = torch.cat(
                (torch.stack(pixel_values), torch.stack(pixel_values_goal)), dim=1
            )

        actions = torch.stack([inst["actions"] for inst in instances])
        goal_pose = torch.stack([inst["goal_pose"] for inst in instances])

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            actions=actions,
            goal_pose=goal_pose,
        )

    def _run_forward_pass(self, batch: dict) -> Tuple[Tensor, Tensor]:
        """Run OmniVLA forward pass and return predicted actions and modality_id."""
        from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

        device_id = self._device_id
        modality_id = self._get_modality_id()

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device_id).startswith("cuda")):
            output = self._vla(
                input_ids=batch["input_ids"].to(device_id),
                attention_mask=batch["attention_mask"].to(device_id),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                modality_id=modality_id.to(torch.bfloat16).to(device_id),
                labels=batch["labels"].to(device_id),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                proprio_projector=self._pose_projector,
                noisy_actions=None,
                noisy_action_projector=None,
                diffusion_timestep_embeddings=None,
                use_film=self.config.use_film,
            )

        # Extract hidden states for action prediction
        ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, self._num_patches:-1]
        batch_size = batch["input_ids"].shape[0]
        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, OMNIVLA_NUM_ACTIONS_CHUNK * OMNIVLA_ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            predicted_actions = self._action_head.predict_action(
                actions_hidden_states,
                modality_id.to(torch.bfloat16).to(device_id),
            )

        return predicted_actions, modality_id

    def _convert_waypoints_to_lekiwi_actions(self, waypoints: Tensor) -> Tensor:
        """Convert OmniVLA waypoint chunk to LeKiwi base velocity chunk.

        Matches run_omnivla.py behavior by selecting one waypoint index
        (config.waypoint_select), converting it with the PD controller, and
        expanding the resulting base command across the returned chunk.
        """
        batch_size, chunk_size, _ = waypoints.shape
        lekiwi_actions = torch.empty(
            (batch_size, chunk_size, LEKIWI_ACTION_DIM),
            dtype=waypoints.dtype,
            device=waypoints.device,
        )

        waypoint_idx = min(max(int(self.config.waypoint_select), 0), chunk_size - 1)
        for batch_idx in range(batch_size):
            linear_vel, angular_vel = self._waypoint_to_limited_velocity(
                waypoints[batch_idx, waypoint_idx].detach().cpu().numpy()
            )
            # LeKiwi base action format: {x.vel, y.vel, theta.vel}
            lekiwi_actions[batch_idx, :, 0] = linear_vel
            lekiwi_actions[batch_idx, :, 1] = 0.0
            lekiwi_actions[batch_idx, :, 2] = angular_vel

        return lekiwi_actions

    def _waypoint_to_limited_velocity(self, waypoint: np.ndarray) -> tuple[float, float]:
        """Apply run_omnivla.py PD controller + velocity limit to one waypoint."""
        waypoint = waypoint.copy()
        waypoint[:2] *= self.config.metric_waypoint_spacing
        dx, dy, hx, hy = waypoint

        eps = 1e-8
        dt = 1 / 3

        if abs(dx) < eps and abs(dy) < eps:
            linear_vel = 0.0
            angular_vel = _clip_angle(math.atan2(hy, hx)) / dt
        elif abs(dx) < eps:
            linear_vel = 0.0
            angular_vel = float(np.sign(dy)) * math.pi / (2 * dt)
        else:
            linear_vel = dx / dt
            angular_vel = math.atan(dy / dx) / dt

        linear_vel = float(np.clip(linear_vel, 0.0, 0.5))
        angular_vel = float(np.clip(angular_vel, -1.0, 1.0))
        linear_vel, angular_vel = _limit_velocity(
            linear_vel,
            angular_vel,
            maxv=self.config.max_linear_vel,
            maxw=self.config.max_angular_vel,
        )
        return linear_vel, angular_vel


def _load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    """Load a checkpoint for a given module, handling naming variants."""
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    if not os.path.exists(checkpoint_path) and module_name == "pose_projector":
        checkpoint_path = os.path.join(path, f"proprio_projector--{step}_checkpoint.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Remove DDP prefix if present
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _tensor_to_pil(tensor: Tensor):
    """Convert a (C, H, W) tensor in [0, 1] or [0, 255] range to PIL Image."""
    from PIL import Image
    import numpy as np

    if tensor.dim() == 3:
        arr = tensor.cpu().numpy()
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255)
            arr = arr.astype(np.uint8)
        # CHW -> HWC
        if arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.shape[2] == 1:
            arr = arr.squeeze(2)
        return Image.fromarray(arr, "RGB")
    else:
        raise ValueError(f"Expected 3D tensor (C,H,W), got {tensor.shape}")


def _clip_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def _limit_velocity(lin: float, ang: float, maxv: float, maxw: float) -> tuple[float, float]:
    if abs(lin) <= maxv:
        if abs(ang) <= maxw:
            return lin, ang
        rd = lin / ang
        return maxw * float(np.sign(lin)) * abs(rd), maxw * float(np.sign(ang))

    if abs(ang) <= 0.001:
        return maxv * float(np.sign(lin)), 0.0

    rd = lin / ang
    if abs(rd) >= maxv / maxw:
        return maxv * float(np.sign(lin)), maxv * float(np.sign(ang)) / abs(rd)

    return maxw * float(np.sign(lin)) * abs(rd), maxw * float(np.sign(ang))
