from dataclasses import dataclass, field
from typing import Optional

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("omnivla")
@dataclass
class OmniVLAConfig(PreTrainedConfig):
    """Configuration for OmniVLA policy.

    OmniVLA is a Vision-Language-Action model for mobile robot navigation.
    It outputs trajectory waypoints (x, y, cos_heading, sin_heading) as action chunks.
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 8  # NUM_ACTIONS_CHUNK from OmniVLA
    n_action_steps: int = 1  # execute one waypoint at a time

    # OmniVLA action/pose dimensions
    action_dim: int = 4  # (dx, dy, cos_heading, sin_heading)
    pose_dim: int = 4  # goal pose dimension (rel_y, rel_x, cos, sin)

    # Model paths
    vla_path: str = "omnivla-finetuned-cast"
    resume_step: Optional[int] = 210000

    # Model architecture
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2  # current + goal image
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0

    # Modality flags
    pose_goal: bool = False
    satellite: bool = False
    image_goal: bool = False
    lan_prompt: bool = True

    # Navigation control
    metric_waypoint_spacing: float = 0.1
    waypoint_select: int = 4  # which waypoint in the chunk to use for control
    max_linear_vel: float = 0.3
    max_angular_vel: float = 0.3
    convert_to_lekiwi_action: bool = True  # convert OmniVLA waypoints to (x.vel, y.vel, theta.vel)

    # Language instruction
    language_instruction: str = "Move to plastic bottle"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )

    def validate_features(self) -> None:
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4)

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
