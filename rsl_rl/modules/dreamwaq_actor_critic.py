"""Asymmetric DreamWaQ policy used by Nezha standing."""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .dreamwaq_estimator import DreamWaQEstimator, _activation


def _mlp(input_dim, hidden_dims, output_dim, activation):
    activation_cls = _activation(activation)
    dimensions = [input_dim, *hidden_dims, output_dim]
    layers = []
    for index in range(len(dimensions) - 1):
        layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
        if index < len(dimensions) - 2:
            layers.append(activation_cls())
    return nn.Sequential(*layers)


class DreamWaQActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 history_length=5, latent_dim=16,
                 actor_hidden_dims=(512, 256, 128),
                 critic_hidden_dims=(512, 256, 128),
                 encoder_hidden_dims=(128,), decoder_hidden_dims=(64, 128),
                 activation="elu", init_noise_std=1.0,
                 vae_sigma_min=0.0, vae_sigma_max=5.0,
                 velocity_target_start=46):
        super().__init__()
        self.num_actor_obs = int(num_actor_obs)
        self.num_critic_obs = int(num_critic_obs)
        self.num_actions = int(num_actions)
        self.history_length = int(history_length)
        self.latent_dim = int(latent_dim)
        self.velocity_target_start = int(velocity_target_start)
        self.estimator = DreamWaQEstimator(
            num_actor_obs, history_length, latent_dim, activation,
            encoder_hidden_dims, decoder_hidden_dims,
            vae_sigma_min, vae_sigma_max,
        )
        self.actor = _mlp(
            num_actor_obs + latent_dim + 3, actor_hidden_dims, num_actions,
            activation,
        )
        self.critic = _mlp(
            num_critic_obs, critic_hidden_dims, 1, activation
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_input(self, observations, observation_history, deterministic=False):
        if deterministic:
            latent, velocity = self.estimator.inference(observation_history)
        else:
            (latent, velocity), _ = self.estimator(observation_history)
        # Keep the source DreamWaQ ordering: latent, estimated velocity, current obs.
        return torch.cat((latent, velocity, observations), dim=-1)

    def act(self, observations, observation_history):
        mean = self.actor(self._actor_input(observations, observation_history))
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        return self.distribution.sample()

    def act_inference(self, observations, observation_history):
        actor_input = self._actor_input(
            observations, observation_history, deterministic=True
        )
        return self.actor(actor_input)

    def evaluate(self, critic_observations):
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def velocity_target(self, critic_observations):
        start = self.velocity_target_start
        return critic_observations[:, start:start + 3]

    def reset(self, dones=None):
        pass
