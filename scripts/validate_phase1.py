#!/usr/bin/env python3
"""Static contract checks that do not require Isaac Gym."""
import ast
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load only configuration classes, bypassing legged_gym.envs/__init__.py and Isaac Gym.
pkg = types.ModuleType('legged_gym.envs.base')
pkg.__path__ = []
sys.modules['legged_gym.envs.base'] = pkg
base = load('legged_gym.envs.base.base_config', ROOT / 'legged_gym/envs/base/base_config.py')
robot = load('legged_gym.envs.base.legged_robot_config', ROOT / 'legged_gym/envs/base/legged_robot_config.py')
cfgmod = load('phase1_config', ROOT / 'nezha_stand/config.py')


class Contracts(unittest.TestCase):
    def test_dimensions_and_method(self):
        cfg, ppo = cfgmod.NezhaStandCfg(), cfgmod.NezhaStandCfgPPO()
        self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs, cfg.env.num_actions), (46, 64, 12))
        self.assertEqual((ppo.runner.policy_class_name, ppo.runner.algorithm_class_name), ('ActorCritic', 'PPO'))
        self.assertEqual(ppo.runner_class_name, 'StandRunner')
        self.assertFalse(ppo.runner.resume)
        self.assertEqual((ppo.runner.load_run, ppo.runner.checkpoint), (-1, -1))
        self.assertGreater(cfg.rewards.scales.height, 0.0)
        self.assertEqual(cfg.rewards.scales.torque_balance, -2.0)
        self.assertFalse(hasattr(cfg.rewards.scales, 'pose'))
        self.assertEqual(cfg.evaluation.max_torque_pair_rms_nm, 10.0)

    def test_runtime_default_dof_positions(self):
        cfg = cfgmod.NezhaStandCfg()
        expected_hips = {'FL_hip_joint': 0.1, 'FR_hip_joint': -0.1,
                         'RL_hip_joint': 0.1, 'RR_hip_joint': -0.1}
        for name, value in expected_hips.items():
            self.assertEqual(cfg.init_state.default_joint_angles[name], value)
        for leg in ('FL', 'FR', 'RL', 'RR'):
            self.assertEqual(cfg.init_state.default_joint_angles[f'{leg}_thigh_joint'], 0.925)
            self.assertEqual(cfg.init_state.default_joint_angles[f'{leg}_calf_joint'], -1.85)
            self.assertEqual(cfg.init_state.default_joint_angles[f'{leg}_foot_joint'], 0.0)

    def test_torque_balance_reward_is_implemented(self):
        tree = ast.parse((ROOT / 'nezha_stand/env.py').read_text())
        methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn('_reward_torque_balance', methods)
        self.assertIn('_torque_pair_rms_nm', methods)
        self.assertNotIn('_reward_pose', methods)

    def test_urdf_joint_partition(self):
        urdf = ET.parse(cfgmod.ASSET_FILE).getroot()
        dofs = {j.get('name') for j in urdf.findall('joint') if j.get('type') != 'fixed'}
        self.assertEqual(dofs, set(cfgmod.LEG_NAMES + cfgmod.WHEEL_NAMES))
        self.assertEqual(len(cfgmod.LEG_NAMES), 12)

    def test_python_syntax(self):
        paths = list((ROOT / 'nezha_stand').glob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
        paths += list((ROOT / 'legged_gym').rglob('*.py')) + list((ROOT / 'rsl_rl').rglob('*.py'))
        for path in paths:
            ast.parse(path.read_text(), filename=str(path))

    def test_standalone_components_are_bundled(self):
        required = (
            'legged_gym/envs/base/base_task.py',
            'legged_gym/envs/base/legged_robot.py',
            'legged_gym/utils/task_registry.py',
            'rsl_rl/modules/actor_critic.py',
            'rsl_rl/algorithms/ppo.py',
            'rsl_rl/storage/rollout_storage.py',
            'rsl_rl/runners/on_policy_runner.py',
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_runtime_has_no_lzhmine_path_dependency(self):
        runtime_files = list((ROOT / 'nezha_stand').glob('*.py'))
        runtime_files += [ROOT / 'scripts/train.py', ROOT / 'scripts/evaluate.py',
                          ROOT / 'scripts/check_physics.py']
        for path in runtime_files:
            source = path.read_text()
            self.assertNotIn('LZHMINE_ROOT', source, str(path))

    def test_train_imports_isaacgym_before_training_stack(self):
        source = (ROOT / 'scripts/train.py').read_text()
        self.assertLess(source.index('from isaacgym import gymapi'),
                        source.index('from legged_gym.utils import'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
