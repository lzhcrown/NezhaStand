"""12 policy actions and 16 physical DOFs; no MINE environment behavior."""
import math
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float
from legged_gym.envs.base.legged_robot import LeggedRobot


class NezhaStandEnv(LeggedRobot):
    def _init_buffers(self):
        # The bundled base uses num_actions for gain/torque allocation. Initialize
        # physical buffers at the DOF width, then allocate 12 policy channels.
        policy_width = self.num_actions
        self.num_actions = self.num_dof
        try:
            super()._init_buffers()
        finally:
            self.num_actions = policy_width
        if self.num_dof != 16 or policy_width != 12:
            raise ValueError('Standing requires 16 physical DOFs and 12 policy actions')
        self.leg_indices = torch.tensor(
            [self.dof_names.index(n) for n in self.cfg.asset.leg_dof_names],
            device=self.device, dtype=torch.long)
        if len(set(self.leg_indices.tolist() + self.wheel_indices.tolist())) != 16:
            raise ValueError('Leg/wheel mappings must partition all DOFs')
        self.actions = torch.zeros(self.num_envs, 12, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.start_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.raw_torques = torch.zeros_like(self.torques)
        self.saturation = torch.zeros(self.num_envs, device=self.device)
        self.base_height = self.root_states[:, 2].clone()
        self.failure_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_height_sum = torch.zeros(self.num_envs, device=self.device)
        self.episode_standing_height_steps = torch.zeros(
            self.num_envs, device=self.device
        )

    def _get_noise_scale_vec(self, cfg):
        self.add_noise = cfg.noise.add_noise
        noise = torch.zeros(46, device=self.device)
        s, scale = cfg.noise.noise_scales, cfg.normalization.obs_scales
        noise[:3] = s.ang_vel * scale.ang_vel
        noise[3:6] = s.gravity
        noise[6:18] = s.dof_pos * scale.dof_pos
        noise[18:34] = s.dof_vel * scale.dof_vel
        return noise * cfg.noise.noise_level

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        for i, name in enumerate(self.dof_names):
            props['friction'][i] = 0.025 if 'hip_joint' in name else 0.05
            props['damping'][i] = 0.0
            if 'armature' in props.dtype.names:
                props['armature'][i] = 0.0
        return props

    def _compute_torques(self, actions):
        ids = self.leg_indices
        target = self.default_dof_pos[:, ids] + self.cfg.control.action_scale * actions
        target = torch.maximum(torch.minimum(target, self.dof_pos_limits[ids, 1]),
                               self.dof_pos_limits[ids, 0])
        # Zero wheel velocity reference means damping, NOT mechanical locking.
        self.raw_torques = -self.d_gains * self.dof_vel
        self.raw_torques[:, ids] += self.p_gains[ids] * (target - self.dof_pos[:, ids])
        self.saturation += (self.raw_torques.abs() >= 0.99 * self.torque_limits).float().mean(1)
        return torch.maximum(torch.minimum(self.raw_torques, self.torque_limits), -self.torque_limits)

    def step(self, actions):
        self.saturation.zero_()
        return super().step(actions)

    def _refresh_derived(self):
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_height = self.root_states[:, 2] - self.env_origins[:, 2]

    def _post_physics_step_callback(self):
        self.commands.zero_()
        self._refresh_derived()

    def check_termination(self):
        c = self.cfg.termination
        bad_contact = torch.any(self.contact_forces[:, self.termination_contact_indices].norm(dim=-1)
                                > c.contact_threshold, dim=1)
        self.failure_buf = (bad_contact
                            | (self.projected_gravity[:, 2] > -math.cos(c.max_tilt_rad))
                            | (self.base_height < c.min_height)
                            | ((self.root_states[:, :2] - self.start_xy).norm(dim=1) > c.max_displacement))
        self.time_out_buf = (self.episode_length_buf >= self.max_episode_length) & ~self.failure_buf
        self.reset_buf = self.failure_buf | self.time_out_buf

    def compute_reward(self):
        super().compute_reward()
        self.rew_buf += self.cfg.rewards.termination_cost * self.failure_buf.float()
        self.episode_return += self.rew_buf
        self.episode_height_sum += self.base_height
        self.episode_standing_height_steps += (
            self.base_height >= self.cfg.rewards.height_gate_start
        ).float()

    def _observations(self, noise):
        ids, s = self.leg_indices, self.obs_scales
        clean = torch.cat((self.base_ang_vel * s.ang_vel, self.projected_gravity,
                           (self.dof_pos[:, ids] - self.default_dof_pos[:, ids]) * s.dof_pos,
                           self.dof_vel[:, ids] * s.dof_vel,
                           self.dof_vel[:, self.wheel_indices] * s.dof_vel, self.actions), dim=1)
        critic = torch.cat((clean, self.base_lin_vel * s.lin_vel,
                            (self.base_height - self.cfg.rewards.base_height_target).unsqueeze(1),
                            self.contact_forces[:, self.feet_indices].flatten(1) / 200.0,
                            self.root_states[:, :2] - self.start_xy), dim=1)
        actor = clean + (2 * torch.rand_like(clean) - 1) * self.noise_scale_vec if noise and self.add_noise else clean
        if actor.shape[1] != 46 or critic.shape[1] != 64:
            raise RuntimeError('Observation contract mismatch')
        return actor, critic

    def compute_observations(self):
        self.obs_buf, self.privileged_obs_buf = self._observations(True)

    def compute_termination_observations(self, env_ids):
        # Evaluator consumes pre-reset states, so failed episodes are not hidden.
        g = self.projected_gravity
        foot_contact = (self.contact_forces[:, self.feet_indices, 2]
                        > self.cfg.rewards.contact_threshold).float()
        foot_load_error_rms = (self._normalized_foot_loads() - 0.25).square().mean(1).sqrt()
        contact_foot_speed_rms = (
            (self.feet_vel[:, :, :2].square().sum(2) * foot_contact).sum(1)
            / foot_contact.sum(1).clamp_min(1.0)).sqrt()
        metrics = torch.stack((
            torch.atan2(-g[:, 1], -g[:, 2]),
            torch.asin(g[:, 0].clamp(-1, 1)),
            self.base_height - self.cfg.rewards.base_height_target,
            self.base_lin_vel[:, :2].norm(dim=1),
            self.saturation / self.cfg.control.decimation,
            self._all_feet_contact(),
            (self.root_states[:, :2] - self.start_xy).norm(dim=1),
            self.dof_vel[:, self.wheel_indices].square().mean(1).sqrt(),
            self.episode_length_buf.float() * self.dt,
            self._torque_pair_rms_nm(),
            foot_load_error_rms,
            contact_foot_speed_rms), dim=1)
        self.extras['stand_metrics'] = metrics.clone()
        self.extras['time_outs'] = self.time_out_buf.clone()
        return self._observations(False)[1][env_ids].clone()

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return self.obs_buf, self.privileged_obs_buf

    def reset_idx(self, env_ids):
        if env_ids.numel() == 0:
            return
        completed = env_ids[self.episode_length_buf[env_ids] > 0]
        if completed.numel():
            completed_steps = self.episode_length_buf[completed].float().clamp_min(1.0)
            self.extras['episode'] = {
                'return': self.episode_return[completed].mean().item(),
                'survival': self.time_out_buf[completed].float().mean().item(),
                'base_height_m': (
                    self.episode_height_sum[completed] / completed_steps
                ).mean().item(),
                'standing_height_fraction': (
                    self.episode_standing_height_steps[completed] / completed_steps
                ).mean().item()}
            for name in self.episode_sums:
                self.extras['episode']['reward/' + name] = (
                    self.episode_sums[name][completed] /
                    (self.episode_length_buf[completed] * self.dt)).mean().item()
        self.dof_pos[env_ids] = self.default_dof_pos
        jitter = self.cfg.init_state.joint_jitter
        q = self.default_dof_pos[:, self.leg_indices] + torch_rand_float(
            -jitter, jitter, (len(env_ids), 12), device=self.device)
        self.dof_pos[env_ids[:, None], self.leg_indices[None, :]] = q
        self.dof_vel[env_ids] = 0
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.start_xy[env_ids] = self.root_states[env_ids, :2]
        ids32 = env_ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                             gymtorch.unwrap_tensor(ids32), len(env_ids))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                    gymtorch.unwrap_tensor(ids32), len(env_ids))
        for buf in (self.actions, self.last_actions, self.last_last_actions, self.last_dof_vel,
                    self.episode_return, self.episode_height_sum,
                    self.episode_standing_height_steps, self.episode_length_buf,
                    self.torques, self.raw_torques):
            buf[env_ids] = 0
        self.contact_forces[env_ids] = 0
        for buf in self.episode_sums.values():
            buf[env_ids] = 0
        self.commands[env_ids] = 0
        self._refresh_derived()

    def post_physics_step(self):
        self.extras = {}
        return super().post_physics_step()

    def _reward_upright(self):
        return torch.exp(-torch.sum((self.projected_gravity - self.gravity_vec)**2, dim=1) / 0.05)

    def _reward_height(self):
        # A quadratic penalty keeps a useful gradient far below the target;
        # the old Gaussian reward saturated near zero and allowed crouching.
        normalized_error = (
            (self.base_height - self.cfg.rewards.base_height_target)
            / self.cfg.rewards.height_error_scale
        )
        return normalized_error.square()

    def _standing_height_gate(self):
        """Smoothly unlock positive standing rewards from 0.45 to 0.50 m."""
        start = self.cfg.rewards.height_gate_start
        target = self.cfg.rewards.base_height_target
        return ((self.base_height - start) / (target - start)).clamp(0.0, 1.0)

    def _reward_stationary(self):
        stationary = torch.exp(
            -self.base_lin_vel.square().sum(1) / 0.04
            - self.base_ang_vel.square().sum(1) / 0.25
        )
        return stationary * self._standing_height_gate()

    def _reward_support(self):
        """Reward only when all four wheels carry a non-trivial vertical load."""
        return self._all_feet_contact() * self._standing_height_gate()

    def _all_feet_contact(self):
        """Raw four-wheel contact indicator used by evaluation metrics."""
        return (
            self.contact_forces[:, self.feet_indices, 2]
            > self.cfg.rewards.contact_threshold
        ).all(1).float()

    def _normalized_foot_loads(self):
        """Vertical wheel loads as fractions of the total supported load."""
        vertical_load = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.0)
        return vertical_load / vertical_load.sum(1, keepdim=True).clamp_min(1e-6)

    def _reward_foot_force_balance(self):
        """Penalize unequal vertical loading while remaining independent of mass."""
        return (self._normalized_foot_loads() - 0.25).square().sum(1)

    def _reward_foot_slip(self):
        """Penalize planar motion of wheel centres that are in ground contact."""
        contact = (self.contact_forces[:, self.feet_indices, 2]
                   > self.cfg.rewards.contact_threshold).float()
        return (self.feet_vel[:, :, :2].square().sum(2) * contact).sum(1)

    def _leg_torque_pair_differences(self, normalize):
        """Six leg-pair differences for hip/thigh/calf torque magnitudes."""
        torque = self.torques[:, self.leg_indices].abs().reshape(self.num_envs, 4, 3)
        if normalize:
            limits = self.torque_limits[self.leg_indices].reshape(1, 4, 3).clamp_min(1e-6)
            torque = torque / limits
        return torch.stack((
            torque[:, 0] - torque[:, 1], torque[:, 0] - torque[:, 2],
            torque[:, 0] - torque[:, 3], torque[:, 1] - torque[:, 2],
            torque[:, 1] - torque[:, 3], torque[:, 2] - torque[:, 3]), dim=1)

    def _reward_torque_balance(self):
        # Average the squared difference over all six leg pairs.  Corresponding
        # hip/thigh/calf joints are compared by normalized magnitude, so mirror
        # signs and different actuator limits do not create false imbalance.
        return self._leg_torque_pair_differences(normalize=True).square().sum(2).mean(1)

    def _torque_pair_rms_nm(self):
        return self._leg_torque_pair_differences(normalize=False).square().mean((1, 2)).sqrt()

    def _reward_wheel_speed(self):
        return self.dof_vel[:, self.wheel_indices].square().sum(1)

    def _reward_drift(self):
        return (self.root_states[:, :2] - self.start_xy).square().sum(1)

    def _reward_dof_vel(self):
        return self.dof_vel[:, self.leg_indices].square().sum(1)

    def _reward_dof_pos_limits(self):
        q = self.dof_pos[:, self.leg_indices]
        limits = self.dof_pos_limits[self.leg_indices]
        return ((limits[:, 0] - q).clamp(min=0) + (q - limits[:, 1]).clamp(min=0)).sum(1)
