"""Runtime adapter for the history-aware DreamWaQ standing policy."""

from typing import Dict, Sequence, Tuple

import torch


NUM_DOFS = 16
NUM_LEG_JOINTS = 12
NUM_OBSERVATIONS = 46
HISTORY_LENGTH = 5
NUM_HISTORY_OBSERVATIONS = NUM_OBSERVATIONS * HISTORY_LENGTH


def _tensor(values, device):
    return torch.as_tensor(values, dtype=torch.float32, device=device)


class NezhaStandPolicyRuntime:
    """Build actor observations and reproduce the Isaac Gym PD controller."""

    def __init__(
        self,
        policy_path: str,
        default_dof_pos: Sequence[float],
        p_gains: Sequence[float],
        d_gains: Sequence[float],
        torque_limits: Sequence[float],
        dof_lower_limits: Sequence[float],
        dof_upper_limits: Sequence[float],
        leg_joint_indices: Sequence[int],
        wheel_joint_indices: Sequence[int],
        action_scale: float = 0.15,
        clip_actions: float = 3.0,
        observation_scales: Dict[str, float] = None,
        history_length: int = HISTORY_LENGTH,
        device: str = "cpu",
    ):
        vector_fields = {
            "default_dof_pos": default_dof_pos,
            "p_gains": p_gains,
            "d_gains": d_gains,
            "torque_limits": torque_limits,
            "dof_lower_limits": dof_lower_limits,
            "dof_upper_limits": dof_upper_limits,
        }
        for name, values in vector_fields.items():
            if len(values) != NUM_DOFS:
                raise ValueError(f"{name} must contain {NUM_DOFS} values")
        if len(leg_joint_indices) != NUM_LEG_JOINTS:
            raise ValueError("leg_joint_indices must contain 12 indices")
        if len(wheel_joint_indices) != 4:
            raise ValueError("wheel_joint_indices must contain 4 indices")
        partition = list(leg_joint_indices) + list(wheel_joint_indices)
        if sorted(partition) != list(range(NUM_DOFS)):
            raise ValueError("leg and wheel indices must partition all 16 DOFs")

        self.device = torch.device(device)
        self.policy = torch.jit.load(policy_path, map_location=self.device).eval()
        self.default_dof_pos = _tensor(default_dof_pos, self.device)
        self.p_gains = _tensor(p_gains, self.device)
        self.d_gains = _tensor(d_gains, self.device)
        self.torque_limits = _tensor(torque_limits, self.device)
        self.lower_limits = _tensor(dof_lower_limits, self.device)
        self.upper_limits = _tensor(dof_upper_limits, self.device)
        self.leg_ids = torch.as_tensor(
            leg_joint_indices, dtype=torch.long, device=self.device
        )
        self.wheel_ids = torch.as_tensor(
            wheel_joint_indices, dtype=torch.long, device=self.device
        )
        self.action_scale = float(action_scale)
        self.clip_actions = float(clip_actions)
        self.history_length = int(history_length)
        if self.history_length != HISTORY_LENGTH:
            raise ValueError(
                f"Exported Nezha policy expects history_length={HISTORY_LENGTH}"
            )
        self.scales = observation_scales or {
            "ang_vel": 1.0,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
        self.previous_actions = torch.zeros(
            NUM_LEG_JOINTS, dtype=torch.float32, device=self.device
        )
        self.observation_history = torch.zeros(
            NUM_HISTORY_OBSERVATIONS, dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            output = self.policy(
                torch.zeros(1, NUM_OBSERVATIONS, device=self.device),
                torch.zeros(1, NUM_HISTORY_OBSERVATIONS, device=self.device),
            )
        if tuple(output.shape) != (1, NUM_LEG_JOINTS):
            raise RuntimeError(
                "Policy returned "
                f"{tuple(output.shape)} for ([1,46], [1,230]); expected [1,12]"
            )

    def reset(self):
        self.previous_actions.zero_()
        self.observation_history.zero_()

    def _append_history(self, observation):
        """Append an observation after it has been used for the current action."""
        self.observation_history = torch.cat((
            self.observation_history[NUM_OBSERVATIONS:], observation
        ))

    def build_observation(
        self,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        dof_pos: Sequence[float],
        dof_vel: Sequence[float],
    ) -> torch.Tensor:
        angular_velocity = _tensor(base_ang_vel, self.device)
        gravity = _tensor(projected_gravity, self.device)
        position = _tensor(dof_pos, self.device)
        velocity = _tensor(dof_vel, self.device)
        expected = ((3,), (3,), (NUM_DOFS,), (NUM_DOFS,))
        for name, value, shape in zip(
            ("base_ang_vel", "projected_gravity", "dof_pos", "dof_vel"),
            (angular_velocity, gravity, position, velocity),
            expected,
        ):
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")

        observation = torch.cat(
            (
                angular_velocity * self.scales["ang_vel"],
                gravity,
                (position[self.leg_ids] - self.default_dof_pos[self.leg_ids])
                * self.scales["dof_pos"],
                velocity[self.leg_ids] * self.scales["dof_vel"],
                velocity[self.wheel_ids] * self.scales["dof_vel"],
                self.previous_actions,
            )
        )
        if tuple(observation.shape) != (NUM_OBSERVATIONS,):
            raise RuntimeError("Standing observation contract is not 46 dimensional")
        return observation

    def compute_torques(
        self,
        actions: torch.Tensor,
        dof_pos: Sequence[float],
        dof_vel: Sequence[float],
    ) -> torch.Tensor:
        actions = _tensor(actions, self.device).reshape(-1)
        if tuple(actions.shape) != (NUM_LEG_JOINTS,):
            raise ValueError("actions must have shape [12]")
        actions = actions.clamp(-self.clip_actions, self.clip_actions)
        position = _tensor(dof_pos, self.device)
        velocity = _tensor(dof_vel, self.device)

        leg_targets = (
            self.default_dof_pos[self.leg_ids] + self.action_scale * actions
        )
        leg_targets = torch.maximum(
            torch.minimum(leg_targets, self.upper_limits[self.leg_ids]),
            self.lower_limits[self.leg_ids],
        )
        # This exactly mirrors NezhaStandEnv._compute_torques: all joints receive
        # viscous damping, and only the 12 leg joints receive position feedback.
        torques = -self.d_gains * velocity
        torques[self.leg_ids] += self.p_gains[self.leg_ids] * (
            leg_targets - position[self.leg_ids]
        )
        return torch.maximum(
            torch.minimum(torques, self.torque_limits), -self.torque_limits
        )

    @torch.inference_mode()
    def step(
        self,
        base_ang_vel: Sequence[float],
        projected_gravity: Sequence[float],
        dof_pos: Sequence[float],
        dof_vel: Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observation = self.build_observation(
            base_ang_vel, projected_gravity, dof_pos, dof_vel
        )
        actions = self.policy(
            observation.unsqueeze(0), self.observation_history.unsqueeze(0)
        ).squeeze(0)
        actions = actions.clamp(-self.clip_actions, self.clip_actions)
        torques = self.compute_torques(actions, dof_pos, dof_vel)
        self._append_history(observation)
        self.previous_actions.copy_(actions)
        return actions, torques
