"""Plain PPO runner adapter for the bundled training stack."""
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class StandRunner(OnPolicyRunner):
    """Construct the regular ActorCritic used by the standalone standing task."""
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg['runner']
        self.alg_cfg = train_cfg['algorithm']
        self.policy_cfg = train_cfg['policy']
        self.device = device
        self.env = env
        critic_width = env.num_privileged_obs or env.num_obs
        actor_critic = ActorCritic(
            env.num_obs, critic_width, env.num_actions, **self.policy_cfg
        ).to(device)
        self.alg = PPO(actor_critic, device=device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg['num_steps_per_env']
        self.save_interval = self.cfg['save_interval']
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env,
            [env.num_obs], [env.num_privileged_obs], [env.num_actions]
        )
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        env.reset()
