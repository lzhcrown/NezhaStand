#!/usr/bin/env python3
"""Train Nezha standing with DreamWaQ history encoding and asymmetric PPO."""
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Isaac Gym must be loaded before any dependency imports PyTorch.
from isaacgym import gymapi  # noqa: F401

from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import get_load_path, update_cfg_from_args
from nezha_stand import TASK_NAME, register_task
from nezha_stand.runner import DreamWaQStandRunner


def _wandb_options():
    return [
        {"name": "--wandb", "action": "store_true", "default": False,
         "help": "Log scalar training curves to Weights & Biases."},
        {"name": "--wandb_project", "type": str, "default": "nezha-stand",
         "help": "W&B project name."},
        {"name": "--wandb_entity", "type": str,
         "help": "Optional W&B user or team name."},
        {"name": "--wandb_group", "type": str,
         "help": "Optional W&B group; defaults to the payload group."},
        {"name": "--wandb_name", "type": str,
         "help": "Optional W&B display name; defaults to the local run name."},
        {"name": "--wandb_tags", "type": str, "default": "",
         "help": "Comma-separated extra W&B tags."},
        {"name": "--wandb_notes", "type": str,
         "help": "Optional W&B run notes."},
        {"name": "--wandb_mode", "type": str, "default": "online",
         "help": "W&B mode: online, offline, or disabled."},
        {"name": "--wandb_run_id", "type": str,
         "help": "Existing W&B run ID when resuming that remote run."},
    ]


def _init_wandb(args, payload_mass, seed, run_name, log_root):
    """Initialize scalar-only W&B logging after explicit --wandb opt-in."""
    if not args.wandb:
        return None
    if args.wandb_mode not in ("online", "offline", "disabled"):
        raise ValueError("--wandb_mode must be online, offline, or disabled")
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging was requested but wandb is not installed. Run: "
            "uv pip install --no-build-isolation -e '.[wandb]'"
        ) from exc

    payload_tag = f"payload-{payload_mass:g}kg"
    automatic_tags = [
        "dreamwaq", "standing", "domain-randomization", payload_tag,
        f"seed-{seed}",
    ]
    extra_tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    tags = list(dict.fromkeys(automatic_tags + extra_tags))
    group = args.wandb_group or payload_tag
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=args.wandb_name or run_name,
        notes=args.wandb_notes,
        tags=tags,
        job_type="train",
        dir=str(log_root),
        mode=args.wandb_mode,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        settings=wandb.Settings(
            console="off",
            disable_code=True,
            disable_git=True,
            x_disable_meta=True,
            x_disable_stats=True,
        ),
    )
    print(f"W&B | project={args.wandb_project} | group={group} | "
          f"run={run.name} | id={run.id} | mode={args.wandb_mode}")
    return run


def train(args):
    register_task()
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    log_root = PROJECT_ROOT / 'logs'
    log_dir = log_root / (datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    log_root.mkdir(parents=True, exist_ok=True)
    wandb_run = _init_wandb(
        args, float(env_cfg.asset.payload_mass_kg), int(train_cfg.seed),
        log_dir.name, log_root,
    )
    runner = None
    try:
        env, env_cfg = task_registry.make_env(TASK_NAME, args=args, env_cfg=env_cfg)
        runner = DreamWaQStandRunner(
            env,
            train_cfg.to_dict() if hasattr(train_cfg, 'to_dict') else _to_dict(train_cfg),
            str(log_dir), args.rl_device, external_logger=wandb_run,
        )
        if train_cfg.runner.resume:
            path = get_load_path(str(log_root), train_cfg.runner.load_run, train_cfg.runner.checkpoint)
            print(f'Loading checkpoint: {path}')
            runner.load(path)
        print(f'DreamWaQ | asset={env_cfg.asset.file} | '
              f'payload={env_cfg.asset.payload_mass_kg:g} kg | obs={env.num_obs} | '
              f'history={env_cfg.env.num_observation_history} | '
              f'critic={env.num_privileged_obs} | actions={env.num_actions} | log={log_dir}')
        print('Domain randomization | '
              f'friction={env_cfg.domain_rand.friction_range} | '
              f'motor_strength={env_cfg.domain_rand.motor_strength_range} | '
              f'payload_mass_kg={env_cfg.domain_rand.payload_mass_range} | '
              'sampling=payload_per_environment,friction_motor_per_episode')
        runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)
    finally:
        if runner is not None:
            runner.close()
        if wandb_run is not None:
            wandb_run.finish()


def _to_dict(obj):
    from legged_gym.utils.helpers import class_to_dict
    return class_to_dict(obj)


if __name__ == '__main__':
    train(get_args(_wandb_options()))
