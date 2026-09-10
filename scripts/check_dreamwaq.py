#!/usr/bin/env python3
"""CPU smoke test for DreamWaQ network, storage and one PPO update."""

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rsl_rl.algorithms import DreamWaQPPO
from rsl_rl.modules import DreamWaQActorCritic


def main():
    environments, steps = 8, 4
    model = DreamWaQActorCritic(
        46, 64, 12, history_length=5, latent_dim=16,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[128], decoder_hidden_dims=[64, 128],
        velocity_target_start=46,
    )
    algorithm = DreamWaQPPO(
        model, num_learning_epochs=1, num_mini_batches=2
    )
    algorithm.init_storage(environments, steps, 46, 64, 230, 12)
    for _ in range(steps):
        observation = torch.randn(environments, 46)
        critic_observation = torch.randn(environments, 64)
        history = torch.randn(environments, 230)
        action = algorithm.act(observation, critic_observation, history)
        algorithm.process_env_step(
            torch.randn(environments),
            torch.zeros(environments, dtype=torch.bool),
            {},
        )
    algorithm.compute_returns(torch.randn(environments, 64))
    losses = algorithm.update()
    if tuple(action.shape) != (environments, 12):
        raise RuntimeError(f"Unexpected action shape: {tuple(action.shape)}")
    if not all(torch.isfinite(torch.tensor(value)) for value in losses.values()):
        raise RuntimeError(f"Non-finite loss: {losses}")
    deterministic = model.act_inference(
        torch.zeros(1, 46), torch.zeros(1, 230)
    )
    if tuple(deterministic.shape) != (1, 12):
        raise RuntimeError("Deterministic DreamWaQ inference contract failed")
    print("DreamWaQ smoke test: OK")
    print("contract: history 5x46 -> latent 16 + velocity 3; actor 65->12")
    print("losses:", {key: round(value, 6) for key, value in losses.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
