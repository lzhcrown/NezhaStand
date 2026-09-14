#!/usr/bin/env python3
"""Run the nominal PD stand with exactly zero policy actions."""
import sys
from pathlib import Path

from isaacgym import gymapi  # noqa: F401; Isaac Gym must be imported before torch.
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legged_gym.utils import get_args, task_registry
from nezha_stand import TASK_NAME, register_task


def check(args):
    register_task()
    args.task = TASK_NAME
    cfg, _ = task_registry.get_cfgs(TASK_NAME)
    cfg.env.num_envs = args.num_envs or 4
    cfg.noise.add_noise = False
    cfg.init_state.joint_jitter = 0.0
    env, _ = task_registry.make_env(TASK_NAME, args=args, env_cfg=cfg)
    steps = int(args.duration / env.dt)
    failures = 0
    height_sum = speed_sum = tilt_sum = wheel_sum = 0.0
    samples = 0
    for i in range(steps):
        with torch.inference_mode():
            _, _, _, dones, infos, _, _ = env.step(
                torch.zeros(env.num_envs, env.num_actions, device=env.device))
        failures += int((dones.bool() & ~infos['time_outs']).sum().item())
        if (i + 1) * env.dt >= cfg.evaluation.settling_time_s:
            m = infos['stand_metrics']
            height_sum += m[:, 2].abs().sum().item()
            speed_sum += m[:, 3].sum().item()
            tilt_sum += torch.maximum(m[:, 0].abs(), m[:, 1].abs()).sum().item()
            wheel_sum += m[:, 7].sum().item()
            samples += env.num_envs
    print(f'zero-action failures: {failures}')
    print(f'mean height error:    {height_sum/max(samples, 1):.5f} m')
    print(f'mean XY speed:        {speed_sum/max(samples, 1):.5f} m/s')
    print(f'mean max tilt:        {tilt_sum/max(samples, 1)*57.29578:.5f} deg')
    print(f'mean wheel RMS:       {wheel_sum/max(samples, 1):.5f} rad/s')
    return 0 if failures == 0 else 2


if __name__ == '__main__':
    extra = [{'name': '--duration', 'type': float, 'default': 5.0,
              'help': 'Simulation duration in seconds.'}]
    raise SystemExit(check(get_args(extra)))
