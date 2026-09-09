#!/usr/bin/env python3
"""Train phase-1 Nezha standing with plain asymmetric PPO."""
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
from nezha_stand.runner import StandRunner


def train(args):
    register_task()
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    env, env_cfg = task_registry.make_env(TASK_NAME, args=args, env_cfg=env_cfg)
    log_root = PROJECT_ROOT / 'logs'
    log_dir = log_root / (datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
    runner = StandRunner(env, train_cfg.to_dict() if hasattr(train_cfg, 'to_dict') else _to_dict(train_cfg),
                         str(log_dir), args.rl_device)
    if train_cfg.runner.resume:
        path = get_load_path(str(log_root), train_cfg.runner.load_run, train_cfg.runner.checkpoint)
        print(f'Loading checkpoint: {path}')
        runner.load(path)
    print(f'Phase 1 | asset={env_cfg.asset.file} | obs={env.num_obs} | '
          f'critic={env.num_privileged_obs} | actions={env.num_actions} | log={log_dir}')
    runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)


def _to_dict(obj):
    from legged_gym.utils.helpers import class_to_dict
    return class_to_dict(obj)


if __name__ == '__main__':
    train(get_args())
