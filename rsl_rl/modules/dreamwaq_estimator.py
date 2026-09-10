"""DreamWaQ history encoder and variational state estimator."""

import torch
import torch.nn as nn
from torch.nn import functional as F


def _activation(name):
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]


class DreamWaQEstimator(nn.Module):
    """Encode observation history into terrain/state latent and body velocity."""

    def __init__(self, num_obs, history_length, latent_dim=16,
                 activation="elu", encoder_hidden_dims=(128,),
                 decoder_hidden_dims=(64, 128), sigma_min=0.0,
                 sigma_max=5.0):
        super().__init__()
        self.num_obs = int(num_obs)
        self.history_length = int(history_length)
        self.latent_dim = int(latent_dim)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        activation_cls = _activation(activation)

        encoder_dims = [self.num_obs * self.history_length,
                        *encoder_hidden_dims, self.latent_dim * 4]
        encoder = []
        for index in range(len(encoder_dims) - 1):
            encoder.append(nn.Linear(encoder_dims[index], encoder_dims[index + 1]))
            if index < len(encoder_dims) - 2:
                encoder.append(activation_cls())
        self.encoder = nn.Sequential(*encoder)
        encoded_dim = self.latent_dim * 4
        self.latent_mu = nn.Linear(encoded_dim, self.latent_dim)
        self.latent_logvar = nn.Linear(encoded_dim, self.latent_dim)
        self.velocity_mu = nn.Linear(encoded_dim, 3)
        self.velocity_logvar = nn.Linear(encoded_dim, 3)

        decoder_dims = [self.latent_dim + 3, *decoder_hidden_dims, self.num_obs]
        decoder = []
        for index in range(len(decoder_dims) - 1):
            decoder.append(nn.Linear(decoder_dims[index], decoder_dims[index + 1]))
            if index < len(decoder_dims) - 2:
                decoder.append(activation_cls())
        self.decoder = nn.Sequential(*decoder)

    def _constrain_logvar(self, logvar):
        sigma = torch.exp(0.5 * logvar)
        sigma = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)
        return 2.0 * torch.log(sigma + 1e-8)

    def encode(self, observation_history):
        encoded = self.encoder(observation_history)
        return (
            self.latent_mu(encoded),
            self._constrain_logvar(self.latent_logvar(encoded)),
            self.velocity_mu(encoded),
            self._constrain_logvar(self.velocity_logvar(encoded)),
        )

    @staticmethod
    def _sample(mean, logvar):
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def forward(self, observation_history):
        latent_mu, latent_logvar, velocity_mu, velocity_logvar = self.encode(
            observation_history
        )
        latent = self._sample(latent_mu, latent_logvar)
        velocity = self._sample(velocity_mu, velocity_logvar)
        return (latent, velocity), (
            latent_mu, latent_logvar, velocity_mu, velocity_logvar
        )

    def inference(self, observation_history):
        latent_mu, _, velocity_mu, _ = self.encode(observation_history)
        return latent_mu, velocity_mu

    def decode(self, latent, velocity):
        return self.decoder(torch.cat((latent, velocity), dim=-1))

    def loss(self, observation_history, observation_target, velocity_target,
             kl_weight=0.1):
        (latent, velocity), params = self(observation_history)
        latent_mu, latent_logvar, _, _ = params
        reconstruction = self.decode(latent, velocity_target)
        reconstruction_loss = F.mse_loss(
            reconstruction, observation_target, reduction="none"
        ).mean(dim=-1)
        velocity_loss = F.mse_loss(
            velocity, velocity_target, reduction="none"
        ).mean(dim=-1)
        kl_loss = -0.5 * torch.sum(
            1.0 + latent_logvar - latent_mu.square() - latent_logvar.exp(), dim=-1
        )
        total = reconstruction_loss + velocity_loss + kl_weight * kl_loss
        return {
            "loss": total,
            "reconstruction": reconstruction_loss,
            "velocity": velocity_loss,
            "kl": kl_loss,
        }
