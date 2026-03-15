"""Tests for OmniVLA policy integration with LeRobot.

These tests verify that OmniVLA is properly registered in LeRobot's
policy system and that the policy interface works correctly for
async inference.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config


class TestOmniVLAConfig:
    """Test OmniVLA configuration."""

    def test_config_creation(self):
        """Test that OmniVLAConfig can be created with defaults."""
        config = make_policy_config("omnivla")
        assert config.chunk_size == 8
        assert config.action_dim == 4
        assert config.pose_dim == 4
        assert config.n_obs_steps == 1
        assert config.n_action_steps == 1
        assert config.convert_to_lekiwi_action is True

    def test_config_custom_params(self):
        """Test OmniVLAConfig with custom parameters."""
        config = make_policy_config(
            "omnivla",
            vla_path="/some/path",
            waypoint_select=3,
            max_linear_vel=0.5,
            convert_to_lekiwi_action=False,
            image_goal=True,
            lan_prompt=True,
            language_instruction="go to the door",
        )
        assert config.vla_path == "/some/path"
        assert config.waypoint_select == 3
        assert config.max_linear_vel == 0.5
        assert config.convert_to_lekiwi_action is False
        assert config.image_goal is True
        assert config.lan_prompt is True
        assert config.language_instruction == "go to the door"

    def test_config_validation(self):
        """Test that n_action_steps > chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="n_action_steps"):
            make_policy_config("omnivla", n_action_steps=10, chunk_size=8)

    def test_config_properties(self):
        """Test config abstract property implementations."""
        config = make_policy_config("omnivla")
        assert config.observation_delta_indices == [0]
        assert config.action_delta_indices == list(range(8))
        assert config.reward_delta_indices is None


class TestOmniVLAPolicyRegistration:
    """Test that OmniVLA is properly registered in LeRobot."""

    def test_get_policy_class(self):
        """Test that OmniVLA policy class can be retrieved by name."""
        policy_class = get_policy_class("omnivla")
        assert policy_class is not None
        assert policy_class.name == "omnivla"

    def test_policy_class_attributes(self):
        """Test that OmniVLA policy class has required attributes."""
        from lerobot.policies.omnivla.modeling_omnivla import OmniVLAPolicy
        from lerobot.policies.omnivla.configuration_omnivla import OmniVLAConfig

        assert OmniVLAPolicy.name == "omnivla"
        assert OmniVLAPolicy.config_class is OmniVLAConfig

    def test_policy_is_pretrained_policy(self):
        """Test that OmniVLAPolicy is a subclass of PreTrainedPolicy."""
        from lerobot.policies.omnivla.modeling_omnivla import OmniVLAPolicy
        from lerobot.policies.pretrained import PreTrainedPolicy

        assert issubclass(OmniVLAPolicy, PreTrainedPolicy)


class TestOmniVLAPolicyInterface:
    """Test OmniVLA policy interface without loading actual model weights."""

    @pytest.fixture
    def policy(self):
        """Create a policy instance without loading models."""
        from lerobot.policies.omnivla.modeling_omnivla import OmniVLAPolicy

        config = make_policy_config("omnivla")
        policy = OmniVLAPolicy.__new__(OmniVLAPolicy)
        # Minimal init without loading models
        policy.config = config
        policy._vla = None
        policy._action_head = None
        policy._pose_projector = None
        policy._action_tokenizer = None
        policy._processor = None
        policy._device_id = None
        policy._num_patches = None
        policy._models_loaded = False
        policy._action_queue = __import__("collections").deque(maxlen=config.n_action_steps)
        return policy

    def test_reset(self, policy):
        """Test that reset clears the action queue."""
        # Add a dummy action
        policy._action_queue.append(torch.zeros(1, 4))
        assert len(policy._action_queue) == 1
        policy.reset()
        assert len(policy._action_queue) == 0

    def test_modality_id_image_goal(self, policy):
        """Test modality ID for image goal only."""
        policy.config.satellite = False
        policy.config.lan_prompt = False
        policy.config.pose_goal = False
        policy.config.image_goal = True
        modality = policy._get_modality_id()
        assert modality.item() == 6

    def test_modality_id_satellite_only(self, policy):
        """Test modality ID for satellite only."""
        policy.config.satellite = True
        policy.config.lan_prompt = False
        policy.config.pose_goal = False
        policy.config.image_goal = False
        modality = policy._get_modality_id()
        assert modality.item() == 0

    def test_modality_id_language_only(self, policy):
        """Test modality ID for language only."""
        policy.config.satellite = False
        policy.config.lan_prompt = True
        policy.config.pose_goal = False
        policy.config.image_goal = False
        modality = policy._get_modality_id()
        assert modality.item() == 7

    def test_modality_id_all_modalities(self, policy):
        """Test modality ID with satellite + pose + image."""
        policy.config.satellite = True
        policy.config.lan_prompt = False
        policy.config.pose_goal = True
        policy.config.image_goal = True
        modality = policy._get_modality_id()
        assert modality.item() == 3

    def test_convert_waypoints_to_lekiwi_actions(self, policy):
        """Waypoint chunk is converted to (x.vel, y.vel, theta.vel) chunk."""
        waypoints = torch.tensor(
            [[[0.5, 0.2, 1.0, 0.0], [0.1, -0.1, 0.0, 1.0]]],
            dtype=torch.float32,
        )
        actions = policy._convert_waypoints_to_lekiwi_actions(waypoints)
        assert actions.shape == (1, 2, 3)
        assert torch.allclose(actions[..., 1], torch.zeros_like(actions[..., 1]))
        # limited by config defaults (0.3, 0.3)
        assert torch.all(actions[..., 0] <= policy.config.max_linear_vel + 1e-6)
        assert torch.all(actions[..., 2].abs() <= policy.config.max_angular_vel + 1e-6)


class TestOmniVLAAsyncInference:
    """Test that OmniVLA works with async inference system."""

    def test_in_supported_policies(self):
        """Test that omnivla is in SUPPORTED_POLICIES list."""
        from lerobot.async_inference.constants import SUPPORTED_POLICIES

        assert "omnivla" in SUPPORTED_POLICIES

    def test_predict_action_chunk_interface(self):
        """Test that predict_action_chunk has the expected signature."""
        from lerobot.policies.omnivla.modeling_omnivla import OmniVLAPolicy
        import inspect

        sig = inspect.signature(OmniVLAPolicy.predict_action_chunk)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "batch" in params

    def test_select_action_interface(self):
        """Test that select_action has the expected signature."""
        from lerobot.policies.omnivla.modeling_omnivla import OmniVLAPolicy
        import inspect

        sig = inspect.signature(OmniVLAPolicy.select_action)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "batch" in params
