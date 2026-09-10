"""Rollout storage carrying the observation history required by DreamWaQ."""

import torch


class DreamWaQRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.observation_history = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_steps, obs_dim, critic_obs_dim,
                 history_dim, action_dim, device="cpu"):
        shape = (num_steps, num_envs)
        self.observations = torch.zeros(*shape, obs_dim, device=device)
        self.critic_observations = torch.zeros(
            *shape, critic_obs_dim, device=device
        )
        self.observation_history = torch.zeros(
            *shape, history_dim, device=device
        )
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.dones = torch.zeros(*shape, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.returns = torch.zeros(*shape, 1, device=device)
        self.advantages = torch.zeros(*shape, 1, device=device)
        self.actions_log_prob = torch.zeros(*shape, 1, device=device)
        self.mu = torch.zeros(*shape, action_dim, device=device)
        self.sigma = torch.zeros(*shape, action_dim, device=device)
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.step = 0

    def add(self, transition):
        if self.step >= self.num_steps:
            raise RuntimeError("DreamWaQ rollout buffer overflow")
        index = self.step
        self.observations[index].copy_(transition.observations)
        self.critic_observations[index].copy_(transition.critic_observations)
        self.observation_history[index].copy_(transition.observation_history)
        self.actions[index].copy_(transition.actions)
        self.rewards[index].copy_(transition.rewards.view(-1, 1))
        self.dones[index].copy_(transition.dones.view(-1, 1).bool())
        self.values[index].copy_(transition.values)
        self.actions_log_prob[index].copy_(
            transition.actions_log_prob.view(-1, 1)
        )
        self.mu[index].copy_(transition.action_mean)
        self.sigma[index].copy_(transition.action_sigma)
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 \
                else self.values[step + 1]
            not_terminal = 1.0 - self.dones[step].float()
            delta = (self.rewards[step] + gamma * not_terminal * next_values
                     - self.values[step])
            advantage = delta + gamma * lam * not_terminal * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages.copy_(self.returns - self.values)
        self.advantages.sub_(self.advantages.mean()).div_(
            self.advantages.std() + 1e-8
        )

    def mini_batches(self, num_mini_batches, num_epochs):
        total = self.num_envs * self.num_steps
        mini_batch_size = total // num_mini_batches
        usable = mini_batch_size * num_mini_batches
        flattened = [
            tensor.flatten(0, 1) for tensor in (
                self.observations, self.critic_observations,
                self.observation_history, self.actions, self.values,
                self.advantages, self.returns, self.actions_log_prob,
                self.mu, self.sigma, self.dones,
            )
        ]
        for _ in range(num_epochs):
            indices = torch.randperm(total, device=self.device)[:usable]
            for start in range(0, usable, mini_batch_size):
                selected = indices[start:start + mini_batch_size]
                yield tuple(tensor[selected] for tensor in flattened)

    def clear(self):
        self.step = 0
