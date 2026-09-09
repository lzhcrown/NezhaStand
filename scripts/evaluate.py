#!/usr/bin/env python3
"""Evaluate a checkpoint against the phase-1 promotion gates."""
import math
import sys
from pathlib import Path

from isaacgym import gymapi  # noqa: F401; Isaac Gym must be imported before torch.
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict, get_load_path
from nezha_stand import TASK_NAME, register_task
from nezha_stand.runner import StandRunner


def evaluate(args):
    register_task()
    args.task = TASK_NAME
    env_cfg, train_cfg = task_registry.get_cfgs(TASK_NAME)
    env_cfg.env.num_envs = args.num_envs or 256
    env_cfg.noise.add_noise = False
    env, _ = task_registry.make_env(TASK_NAME, args=args, env_cfg=env_cfg)
    runner = StandRunner(env, class_to_dict(train_cfg), None, args.rl_device)
    checkpoint = args.checkpoint_path
    if checkpoint is None:
        checkpoint = get_load_path(str(PROJECT_ROOT / 'logs'), -1, -1)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = env.get_observations()
    sums = torch.zeros(6, device=env.device)  # roll2, pitch2, height2, speed2, saturation, support
    samples = completed = successes = 0
    max_drift = 0.0
    wheel_square_sum = 0.0
    torque_pair_square_sum = 0.0
    target_episodes = args.episodes
    settle = env_cfg.evaluation.settling_time_s
    while completed < target_episodes:
        with torch.inference_mode():
            obs, _, _, dones, infos, _, _ = env.step(policy(obs))
        m = infos['stand_metrics']
        valid = m[:, 8] >= settle
        if valid.any():
            v = m[valid]
            sums += torch.stack((v[:, 0].square().sum(), v[:, 1].square().sum(),
                                 v[:, 2].square().sum(), v[:, 3].square().sum(),
                                 v[:, 4].sum(), v[:, 5].sum()))
            wheel_square_sum += v[:, 7].square().sum().item()
            torque_pair_square_sum += v[:, 9].square().sum().item()
            max_drift = max(max_drift, v[:, 6].max().item())
            samples += int(valid.sum())
        done = dones.bool()
        completed += int(done.sum())
        successes += int(infos['time_outs'][done].sum())
    rms = torch.sqrt(sums[:4] / max(samples, 1))
    values = {
        'survival': successes / max(completed, 1),
        'roll_pitch_rms_deg': max(rms[0].item(), rms[1].item()) * 180 / math.pi,
        'height_rmse_m': rms[2].item(), 'xy_speed_rms_m_s': rms[3].item(),
        'saturation_fraction': (sums[4] / max(samples, 1)).item(),
        'all_feet_contact_fraction': (sums[5] / max(samples, 1)).item(),
        'max_xy_drift_m': max_drift,
        'wheel_rms_rad_s': math.sqrt(wheel_square_sum / max(samples, 1)),
        'torque_pair_rms_nm': math.sqrt(torque_pair_square_sum / max(samples, 1)),
    }
    limits = env_cfg.evaluation
    passed = {
        'survival': values['survival'] >= limits.min_survival,
        'roll_pitch_rms_deg': values['roll_pitch_rms_deg'] <= limits.max_roll_pitch_rms_deg,
        'height_rmse_m': values['height_rmse_m'] <= limits.max_height_rmse_m,
        'xy_speed_rms_m_s': values['xy_speed_rms_m_s'] <= limits.max_xy_speed_rms_m_s,
        'saturation_fraction': values['saturation_fraction'] <= limits.max_saturation_fraction,
        'all_feet_contact_fraction': values['all_feet_contact_fraction'] >= limits.min_all_feet_contact_fraction,
        'max_xy_drift_m': values['max_xy_drift_m'] <= limits.max_xy_drift_m,
        'wheel_rms_rad_s': values['wheel_rms_rad_s'] <= limits.max_wheel_rms_rad_s,
        'torque_pair_rms_nm': values['torque_pair_rms_nm'] <= limits.max_torque_pair_rms_nm,
    }
    for name, value in values.items():
        print(f'{name:28s} {value:10.5f}  {"PASS" if passed[name] else "FAIL"}')
    print('PROMOTE_TO_PHASE_2=' + ('YES' if all(passed.values()) else 'NO'))
    return 0 if all(passed.values()) else 2


if __name__ == '__main__':
    params = [
        {'name': '--checkpoint_path', 'type': str, 'help': 'Checkpoint path; default is latest model.'},
        {'name': '--episodes', 'type': int, 'default': 1024, 'help': 'Completed episodes to evaluate.'},
    ]
    raise SystemExit(evaluate(get_args(params)))
