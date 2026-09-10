"""DreamWaQ on-policy runner for the standalone Nezha standing task."""

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import DreamWaQPPO
from rsl_rl.modules import DreamWaQActorCritic


class DreamWaQStandRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.history_length = int(self.policy_cfg["history_length"])
        self.history_dim = env.num_obs * self.history_length
        critic_dim = env.num_privileged_obs or env.num_obs
        self.actor_critic = DreamWaQActorCritic(
            env.num_obs, critic_dim, env.num_actions, **self.policy_cfg
        ).to(device)
        self.alg = DreamWaQPPO(
            self.actor_critic, device=device, **self.alg_cfg
        )
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg["save_interval"])
        self.alg.init_storage(
            env.num_envs, self.num_steps_per_env, env.num_obs, critic_dim,
            self.history_dim, env.num_actions,
        )
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0
        env.reset()

    def initial_history(self):
        """Cold-start history, matching DreamWaQ's zero-filled buffer."""
        return torch.zeros(
            self.env.num_envs, self.history_dim,
            dtype=torch.float, device=self.device,
        )

    @staticmethod
    def update_history(history, observation, dones):
        """Append the pre-action observation and clear terminated environments."""
        obs_dim = observation.shape[-1]
        history = torch.cat((history[:, obs_dim:], observation), dim=-1)
        history[dones.reshape(-1).bool()] = 0.0
        return history

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )
        obs = self.env.get_observations().to(self.device)
        privileged = self.env.get_privileged_observations()
        critic_obs = (privileged if privileged is not None else obs).to(self.device)
        history = self.initial_history()
        self.actor_critic.train()

        episode_infos = []
        reward_buffer = deque(maxlen=100)
        length_buffer = deque(maxlen=100)
        reward_sum = torch.zeros(self.env.num_envs, device=self.device)
        episode_length = torch.zeros(self.env.num_envs, device=self.device)
        best_reward = -float("inf")
        first_iteration = self.current_learning_iteration
        final_iteration = first_iteration + num_learning_iterations

        for iteration in range(first_iteration, final_iteration):
            collection_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, history)
                    old_obs = obs
                    (obs, privileged, rewards, dones, infos, _, _) = self.env.step(actions)
                    obs = obs.to(self.device)
                    critic_obs = (
                        privileged if privileged is not None else obs
                    ).to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    history = self.update_history(history, old_obs, dones)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            episode_infos.append(infos["episode"])
                        reward_sum += rewards
                        episode_length += 1
                        done_ids = dones.reshape(-1).bool()
                        reward_buffer.extend(reward_sum[done_ids].cpu().tolist())
                        length_buffer.extend(episode_length[done_ids].cpu().tolist())
                        reward_sum[done_ids] = 0.0
                        episode_length[done_ids] = 0.0
                collection_time = time.time() - collection_start
                self.alg.compute_returns(critic_obs)

            learning_start = time.time()
            losses = self.alg.update()
            learning_time = time.time() - learning_start
            self.current_learning_iteration = iteration + 1
            self._log(
                iteration, final_iteration, losses, episode_infos,
                reward_buffer, length_buffer, collection_time, learning_time,
            )
            if self.log_dir is not None:
                if iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
                if reward_buffer:
                    mean_reward = statistics.mean(reward_buffer)
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        self.save(os.path.join(self.log_dir, "best_policy.pt"))
            episode_infos.clear()

        if self.log_dir is not None:
            self.save(os.path.join(
                self.log_dir, f"model_{self.current_learning_iteration}.pt"
            ))

    def _log(self, iteration, final_iteration, losses, episode_infos,
             reward_buffer, length_buffer, collection_time, learning_time):
        steps = self.num_steps_per_env * self.env.num_envs
        iteration_time = collection_time + learning_time
        self.tot_timesteps += steps
        self.tot_time += iteration_time
        fps = int(steps / max(iteration_time, 1e-6))
        if self.writer is not None:
            names = {
                "value": "Loss/value_function",
                "surrogate": "Loss/surrogate",
                "vae": "Loss/vae_total",
                "reconstruction": "Loss/vae_reconstruction",
                "velocity": "Loss/vae_velocity",
                "kl": "Loss/vae_kl",
                "entropy": "Loss/entropy",
            }
            for key, tag in names.items():
                self.writer.add_scalar(tag, losses[key], iteration)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
            self.writer.add_scalar(
                "Policy/mean_noise_std", self.actor_critic.std.mean().item(), iteration
            )
            self.writer.add_scalar("Perf/total_fps", fps, iteration)
            if reward_buffer:
                self.writer.add_scalar(
                    "Train/mean_reward", statistics.mean(reward_buffer), iteration
                )
                self.writer.add_scalar(
                    "Train/mean_episode_length", statistics.mean(length_buffer), iteration
                )
            for key in episode_infos[0] if episode_infos else ():
                values = [
                    torch.as_tensor(info[key], device=self.device).float().reshape(-1)
                    for info in episode_infos
                ]
                self.writer.add_scalar(
                    "Episode/" + key, torch.cat(values).mean(), iteration
                )
        summary = (
            f"iteration {iteration}/{final_iteration} | {fps} steps/s | "
            f"value {losses['value']:.4f} | policy {losses['surrogate']:.4f} | "
            f"VAE {losses['vae']:.4f} (recon {losses['reconstruction']:.4f}, "
            f"vel {losses['velocity']:.4f}, KL {losses['kl']:.4f})"
        )
        if reward_buffer:
            summary += f" | reward {statistics.mean(reward_buffer):.2f}"
        print(summary)

    def save(self, path, infos=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "architecture": "DreamWaQ",
            "model_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "vae_optimizer_state_dict": self.alg.vae_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("architecture") != "DreamWaQ":
            raise RuntimeError(
                "This is not a DreamWaQ checkpoint; old plain-PPO checkpoints "
                "cannot be resumed with the history encoder."
            )
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.vae_optimizer.load_state_dict(
                checkpoint["vae_optimizer_state_dict"]
            )
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference


# Backward-compatible import name; this class is no longer plain PPO.
StandRunner = DreamWaQStandRunner
