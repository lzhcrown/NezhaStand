#!/usr/bin/env python3
"""Export a DreamWaQ checkpoint as a history-aware TorchScript policy."""

import argparse
import json
import re
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rsl_rl.modules import DreamWaQActorCritic


class DreamWaQPolicyExporter(torch.nn.Module):
    """Deployable deterministic estimator plus actor."""

    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = actor_critic.estimator.encoder
        self.latent_mu = actor_critic.estimator.latent_mu
        self.velocity_mu = actor_critic.estimator.velocity_mu
        self.actor = actor_critic.actor

    def forward(self, observation, observation_history):
        encoded = self.encoder(observation_history)
        latent = self.latent_mu(encoded)
        velocity = self.velocity_mu(encoded)
        return self.actor(torch.cat((latent, velocity, observation), dim=-1))


def _latest_training_run():
    candidates = [
        path
        for path in (PROJECT_ROOT / "logs").iterdir()
        if path.is_dir() and path.name != "exported"
    ]
    if not candidates:
        raise FileNotFoundError("No training run was found under logs/")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_checkpoint(run_directory):
    numbered = []
    for path in run_directory.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered)[1]
    best = run_directory / "best_policy.pt"
    if best.is_file():
        return best
    raise FileNotFoundError(f"No model_*.pt or best_policy.pt in {run_directory}")


def _resolve_checkpoint(args):
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        run_name = checkpoint.parent.name
    else:
        run_directory = (
            (PROJECT_ROOT / "logs" / args.run).resolve()
            if args.run
            else _latest_training_run()
        )
        checkpoint = _latest_checkpoint(run_directory)
        run_name = run_directory.name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, run_name


def export(args):
    checkpoint_path, run_name = _resolve_checkpoint(args)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PROJECT_ROOT / "logs" / "exported" / "latest" / "policy.pt"
    )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("architecture") != "DreamWaQ":
        raise RuntimeError(
            "Checkpoint is from the old plain-PPO architecture. Train a new "
            "DreamWaQ run before exporting."
        )
    actor_critic = DreamWaQActorCritic(
        46, 64, 12, history_length=5, latent_dim=16,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        encoder_hidden_dims=[128], decoder_hidden_dims=[64, 128],
        activation="elu", init_noise_std=0.5,
        vae_sigma_min=0.0, vae_sigma_max=5.0,
        velocity_target_start=46,
    ).eval()
    actor_critic.load_state_dict(checkpoint["model_state_dict"], strict=True)
    deploy_policy = DreamWaQPolicyExporter(actor_critic).cpu().eval()
    scripted = torch.jit.script(deploy_policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    loaded = torch.jit.load(str(output_path), map_location="cpu").eval()
    with torch.inference_mode():
        result = loaded(torch.zeros(1, 46), torch.zeros(1, 230))
    if tuple(result.shape) != (1, 12) or not torch.isfinite(result).all():
        raise RuntimeError("Exported policy failed ([1,46], [1,230]) -> [1,12]")

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "source_run": run_name,
        "checkpoint_iteration": checkpoint.get("iter"),
        "architecture": "DreamWaQ",
        "actor_observations": 46,
        "history_length": 5,
        "history_observations": 230,
        "latent_dim": 16,
        "actions": 12,
        "action_scale": 0.25,
        "clip_actions": 3.0,
        "control_frequency_hz": 50,
        "training_payload_mass_kg": checkpoint.get("payload_mass_kg", 0.0),
        "p_gains_per_leg": [220.0, 220.0, 300.0, 0.0],
        "d_gains_per_leg": [4.0, 4.0, 4.0, 1.2],
        "policy_file": output_path.name,
    }
    with open(output_path.with_name("policy_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"checkpoint: {checkpoint_path}")
    print(f"policy:     {output_path}")
    print("validated:  inputs [1,46] + [1,230], output [1,12]")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", help="Run directory name under logs/")
    source.add_argument("--checkpoint", help="Explicit model_*.pt or best_policy.pt")
    parser.add_argument("--output", help="Output policy.pt path")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(export(parse_args()))
