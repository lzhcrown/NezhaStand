#!/usr/bin/env python3
"""Check external dependencies and verify that training code resolves locally."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def is_inside_project(path):
    try:
        Path(path).resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def main():
    errors = []
    print(f'Python:    {sys.version.split()[0]}')
    if sys.version_info[:2] != (3, 8):
        errors.append('Isaac Gym Preview 4 environment must use Python 3.8')

    try:
        import numpy as np
        print(f'NumPy:     {np.__version__}')
        if np.__version__ != '1.23.5':
            errors.append(
                'Isaac Gym environment must use numpy==1.23.5; run: '
                'uv pip install --force-reinstall numpy==1.23.5'
            )
    except Exception as exc:
        errors.append(f'NumPy import failed: {exc}')

    try:
        from isaacgym import gymapi  # noqa: F401
        import isaacgym
        print(f'Isaac Gym: OK ({Path(isaacgym.__file__).resolve()})')
    except Exception as exc:
        errors.append(f'Isaac Gym import failed: {exc}')

    try:
        import torch
        print(f'PyTorch:   {torch.__version__}')
        print(f'CUDA:      {torch.cuda.is_available()} ({torch.version.cuda})')
        if not torch.cuda.is_available():
            errors.append('PyTorch cannot access CUDA; GPU training is unavailable')
    except Exception as exc:
        errors.append(f'PyTorch import failed: {exc}')

    for name in ('legged_gym', 'rsl_rl', 'nezha_stand'):
        try:
            module = __import__(name)
            module_path = Path(module.__file__).resolve()
            print(f'{name:11s} {module_path}')
            if not is_inside_project(module_path):
                errors.append(f'{name} resolves outside NezhaStand: {module_path}')
        except Exception as exc:
            errors.append(f'{name} import failed: {exc}')

    try:
        from legged_gym.utils import task_registry  # noqa: F401
        from rsl_rl.algorithms import PPO  # noqa: F401
        from rsl_rl.modules import ActorCritic  # noqa: F401
        from rsl_rl.algorithms import DreamWaQPPO  # noqa: F401
        from rsl_rl.modules import DreamWaQActorCritic  # noqa: F401
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner  # noqa: F401
        from nezha_stand import register_task
        register_task()
        print('Training stack: OK')
    except Exception as exc:
        errors.append(f'training-stack import failed: {exc}')

    asset = PROJECT_ROOT / 'assets/nezha/urdf/nezha.urdf'
    print(f'URDF:      {asset}')
    if not asset.is_file():
        errors.append(f'URDF is missing: {asset}')

    if errors:
        for error in errors:
            print(f'ERROR: {error}', file=sys.stderr)
        return 1
    print('installation: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
