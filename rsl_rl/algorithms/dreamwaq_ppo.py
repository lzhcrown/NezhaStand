"""PPO with the DreamWaQ variational history-estimation objective."""

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import DreamWaQRolloutStorage


class DreamWaQPPO:
    def __init__(self, actor_critic, num_learning_epochs=5,
                 num_mini_batches=4, clip_param=0.2, gamma=0.99, lam=0.95,
                 value_loss_coef=1.0, entropy_coef=0.005,
                 learning_rate=1e-3, vae_learning_rate=1e-3,
                 kl_weight=0.1, max_grad_norm=1.0,
                 use_clipped_value_loss=True, schedule="adaptive",
                 desired_kl=0.01, device="cpu"):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.learning_rate = learning_rate
        self.vae_learning_rate = vae_learning_rate
        self.kl_weight = kl_weight
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl

        # The Go2W source uses two Adam instances over overlapping parameters.
        # Keep the same joint RL+VAE objective, but give each parameter exactly
        # one optimizer to avoid inconsistent Adam moments and double updates.
        estimator_ids = {id(parameter) for parameter in actor_critic.estimator.parameters()}
        policy_parameters = [
            parameter for parameter in actor_critic.parameters()
            if id(parameter) not in estimator_ids
        ]
        self.optimizer = optim.Adam(policy_parameters, lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            actor_critic.estimator.parameters(), lr=vae_learning_rate
        )
        self.storage = None
        self.transition = DreamWaQRolloutStorage.Transition()

    def init_storage(self, num_envs, num_steps, obs_dim, critic_obs_dim,
                     history_dim, action_dim):
        self.storage = DreamWaQRolloutStorage(
            num_envs, num_steps, obs_dim, critic_obs_dim, history_dim,
            action_dim, self.device,
        )

    def act(self, observations, critic_observations, observation_history):
        transition = self.transition
        transition.actions = self.actor_critic.act(
            observations, observation_history
        ).detach()
        transition.values = self.actor_critic.evaluate(
            critic_observations
        ).detach()
        transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            transition.actions
        ).detach()
        transition.action_mean = self.actor_critic.action_mean.detach()
        transition.action_sigma = self.actor_critic.action_std.detach()
        transition.observations = observations
        transition.critic_observations = critic_observations
        transition.observation_history = observation_history
        return transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            time_outs = infos["time_outs"].to(self.device).unsqueeze(1)
            self.transition.rewards += self.gamma * (
                self.transition.values * time_outs
            ).squeeze(1)
        self.storage.add(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_observations):
        last_values = self.actor_critic.evaluate(
            last_critic_observations
        ).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def _adapt_learning_rate(self, mu, sigma, old_mu, old_sigma):
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.square() + (old_mu - mu).square())
                / (2.0 * sigma.square()) - 0.5,
                dim=-1,
            ).mean()
            if kl > 2.0 * self.desired_kl:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl < 0.5 * self.desired_kl:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

    def update(self):
        totals = {
            "value": 0.0, "surrogate": 0.0, "vae": 0.0,
            "entropy": 0.0, "kl": 0.0, "velocity": 0.0,
            "reconstruction": 0.0,
        }
        updates = 0
        for batch in self.storage.mini_batches(
            self.num_mini_batches, self.num_learning_epochs
        ):
            (obs, critic_obs, history, actions, old_values, advantages,
             returns, old_log_prob, old_mu, old_sigma, dones) = batch
            self.actor_critic.act(obs, history)
            log_prob = self.actor_critic.get_actions_log_prob(actions)
            values = self.actor_critic.evaluate(critic_obs)
            mu = self.actor_critic.action_mean
            sigma = self.actor_critic.action_std
            entropy = self.actor_critic.entropy
            self._adapt_learning_rate(mu, sigma, old_mu, old_sigma)

            ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
            surrogate = -advantages.squeeze(-1) * ratio
            clipped = -advantages.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.maximum(surrogate, clipped).mean()
            if self.use_clipped_value_loss:
                clipped_values = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.maximum(
                    (values - returns).square(),
                    (clipped_values - returns).square(),
                ).mean()
            else:
                value_loss = (returns - values).square().mean()
            ppo_loss = (surrogate_loss + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy.mean())

            vae_parts = self.actor_critic.estimator.loss(
                history, obs, self.actor_critic.velocity_target(critic_obs),
                self.kl_weight,
            )
            valid = ~dones.squeeze(-1)
            if not valid.any():
                valid = torch.ones_like(valid)
            vae_loss = vae_parts["loss"][valid].mean()

            self.optimizer.zero_grad()
            self.vae_optimizer.zero_grad()
            (ppo_loss + vae_loss).backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.vae_optimizer.step()

            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate_loss.item()
            totals["vae"] += vae_loss.item()
            totals["entropy"] += entropy.mean().item()
            totals["kl"] += vae_parts["kl"][valid].mean().item()
            totals["velocity"] += vae_parts["velocity"][valid].mean().item()
            totals["reconstruction"] += vae_parts["reconstruction"][valid].mean().item()
            updates += 1
        self.storage.clear()
        return {key: value / max(updates, 1) for key, value in totals.items()}
