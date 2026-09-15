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
payload_math = load('payload_math', ROOT / 'nezha_stand/payload.py')


class Contracts(unittest.TestCase):
    def test_dimensions_and_method(self):
        cfg, ppo = cfgmod.NezhaStandCfg(), cfgmod.NezhaStandCfgPPO()
        self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs, cfg.env.num_actions), (46, 64, 12))
        self.assertEqual(cfg.env.observation_history_length, 5)
        self.assertEqual(cfg.env.num_observation_history, 230)
        self.assertEqual(
            (ppo.runner.policy_class_name, ppo.runner.algorithm_class_name),
            ('DreamWaQActorCritic', 'DreamWaQPPO'),
        )
        self.assertEqual(ppo.runner_class_name, 'DreamWaQStandRunner')
        self.assertEqual(ppo.policy.latent_dim, 16)
        self.assertEqual(ppo.policy.actor_hidden_dims, [512, 256, 128])
        self.assertEqual(ppo.policy.critic_hidden_dims, [512, 256, 128])
        self.assertEqual(ppo.policy.encoder_hidden_dims, [128])
        self.assertEqual(ppo.policy.decoder_hidden_dims, [64, 128])
        self.assertEqual(ppo.policy.velocity_target_start, 46)
        self.assertEqual(ppo.runner.num_steps_per_env, 48)
        self.assertEqual(ppo.algorithm.vae_learning_rate, 1e-3)
        self.assertEqual(ppo.algorithm.kl_weight, 0.1)
        self.assertFalse(ppo.runner.resume)
        self.assertEqual((ppo.runner.load_run, ppo.runner.checkpoint), (-1, -1))
        self.assertEqual(cfg.rewards.base_height_target, 0.50)
        self.assertEqual(cfg.rewards.height_error_scale, 0.10)
        self.assertEqual(cfg.rewards.height_gate_start, 0.40)
        self.assertEqual(cfg.rewards.scales.height, -4.0)
        self.assertEqual(cfg.termination.min_height, 0.30)
        self.assertEqual(cfg.rewards.termination_cost, -10.0)
        self.assertEqual(cfg.rewards.scales.collision, -2.0)
        self.assertEqual(cfg.rewards.scales.torque_balance, -4.0)
        self.assertEqual(cfg.rewards.scales.nominal_pose, -1.0)
        self.assertEqual(cfg.rewards.scales.foot_force_balance, -1.0)
        self.assertEqual(cfg.rewards.max_contact_force, 800.0)
        self.assertEqual(cfg.rewards.contact_threshold, 20.0)
        self.assertGreater(cfg.rewards.scales.support, 0.0)
        self.assertLess(cfg.rewards.scales.foot_force_balance, 0.0)
        self.assertLess(cfg.rewards.scales.foot_slip, 0.0)
        for forbidden in ('pose', 'default_pos', 'default_pos_reward',
                          'feet_air_time', 'feet_clearance', 'feet_on_air'):
            self.assertFalse(hasattr(cfg.rewards.scales, forbidden), forbidden)
        self.assertEqual(cfg.evaluation.max_torque_pair_rms_nm, 10.0)
        self.assertEqual(cfg.evaluation.max_foot_load_fraction_rms, 0.10)
        self.assertEqual(cfg.evaluation.max_contact_foot_speed_rms_m_s, 0.05)

    def test_runtime_default_dof_positions(self):
        cfg = cfgmod.NezhaStandCfg()
        expected_hips = {'FL_hip_joint': -0.1, 'FR_hip_joint': 0.1,
                         'RL_hip_joint': -0.1, 'RR_hip_joint': 0.1}
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
        self.assertIn('_reward_support', methods)
        self.assertIn('_standing_height_gate', methods)
        self.assertIn('_all_feet_contact', methods)
        self.assertIn('_reward_foot_force_balance', methods)
        self.assertIn('_reward_foot_slip', methods)
        self.assertIn('_reward_nominal_pose', methods)
        self.assertNotIn('_reward_pose', methods)

    def test_exactly_four_support_feet(self):
        cfg = cfgmod.NezhaStandCfg()
        self.assertEqual(cfg.asset.feet_names,
                         ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'])
        self.assertEqual(cfg.asset.payload_body_name, 'top_box')
        self.assertGreaterEqual(cfg.asset.payload_mass_kg, 0.0)
        self.assertTrue(cfg.asset.collapse_fixed_joints)

    def test_adopted_standing_gains(self):
        cfg, ppo = cfgmod.NezhaStandCfg(), cfgmod.NezhaStandCfgPPO()
        self.assertEqual(cfg.control.stiffness,
                         {'hip_joint': 220.0, 'thigh_joint': 220.0,
                          'calf_joint': 300.0, 'foot_joint': 0.0})
        self.assertEqual(cfg.control.damping,
                         {'hip_joint': 4.0, 'thigh_joint': 4.0,
                          'calf_joint': 4.0, 'foot_joint': 1.2})
        self.assertEqual(cfg.control.action_scale, 0.25)
        self.assertEqual(cfg.init_state.joint_jitter, 0.01)
        self.assertEqual(cfg.noise.noise_level, 0.25)
        self.assertEqual(ppo.runner.max_iterations, 30000)

    def test_requested_domain_randomization(self):
        cfg = cfgmod.NezhaStandCfg()
        self.assertTrue(cfg.domain_rand.randomize_friction)
        self.assertEqual(cfg.domain_rand.friction_range, [0.8, 1.2])
        self.assertTrue(cfg.domain_rand.randomize_motor_strength)
        self.assertEqual(cfg.domain_rand.motor_strength_range, [0.95, 1.05])
        self.assertTrue(cfg.domain_rand.randomize_payload_mass)
        self.assertEqual(cfg.domain_rand.payload_mass_range, [88.0, 92.0])
        self.assertEqual(cfg.asset.payload_mass_kg, 90.0)
        self.assertEqual(cfg.asset.payload_com_in_carrier, [0.0, -0.03, 0.31])
        self.assertEqual(
            cfg.asset.payload_inertia_per_kg,
            [0.01702, 0.027395, 0.03453],
        )
        source = (ROOT / 'nezha_stand/env.py').read_text()
        self.assertIn('def _process_rigid_body_props', source)
        self.assertNotIn('float(prop.inertia.x)', source)
        self.assertIn('float(matrix.x.x)', source)
        self.assertIn('prop.invInertia', source)
        self.assertIn('self.raw_torques *= self.motor_strength_factors', source)
        self.assertIn('self.motor_strength_factors[env_ids] = torch_rand_float', source)
        self.assertIn('self.refresh_actor_rigid_shape_props(env_ids)', source)

    def test_payload_mass_property_math(self):
        old_mass = 100.0
        old_com = [0.0, 0.0, 0.10]
        old_inertia = [[8.0, 0.1, 0.0], [0.1, 10.0, 0.2], [0.0, 0.2, 12.0]]
        delta_mass = 2.0
        payload_com = [0.0, -0.03, 0.31]
        inertia_per_kg = [0.01702, 0.027395, 0.03453]
        new_mass, new_com, new_inertia = (
            payload_math.update_composite_mass_properties(
                old_mass, old_com, old_inertia, delta_mass,
                payload_com, inertia_per_kg,
            )
        )
        self.assertEqual(new_mass, 102.0)
        self.assertAlmostEqual(new_com[1], -0.06 / 102.0)
        self.assertAlmostEqual(new_com[2], 10.62 / 102.0)
        self.assertTrue(
            payload_math.is_positive_definite_symmetric(new_inertia)
        )
        inverse = payload_math.inverse_matrix3(new_inertia)
        product = [
            [sum(new_inertia[row][k] * inverse[k][column]
                 for k in range(3)) for column in range(3)]
            for row in range(3)
        ]
        for row in range(3):
            for column in range(3):
                self.assertAlmostEqual(
                    product[row][column], 1.0 if row == column else 0.0,
                    places=9,
                )
        reduced_mass, _, reduced_inertia = (
            payload_math.update_composite_mass_properties(
                old_mass, old_com, old_inertia, -2.0,
                payload_com, inertia_per_kg,
            )
        )
        self.assertEqual(reduced_mass, 98.0)
        self.assertTrue(
            payload_math.is_positive_definite_symmetric(reduced_inertia)
        )

    def test_mujoco_contact_observability_is_bundled(self):
        simulator = ROOT / 'mujoco/nezha_stand_sim.py'
        tree = ast.parse(simulator.read_text())
        functions = {node.name for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)}
        self.assertIn('_wheel_vertical_forces', functions)
        self.assertIn('_contact_metrics', functions)
        self.assertIn('_csv_row', functions)
        self.assertIn('_apply_payload_mass', functions)
        self.assertTrue((ROOT / 'scripts/plot_mujoco_contacts.py').is_file())

    def test_mujoco_uses_separate_lzhmine_aligned_environment(self):
        requirements = (ROOT / 'mujoco/requirements.txt').read_text()
        self.assertIn('mujoco==3.12.0', requirements)
        validator = (ROOT / 'scripts/validate_mujoco.py').read_text()
        self.assertIn('sys.version_info[:2] != (3, 11)', validator)
        self.assertIn('mujoco.__version__ != "3.12.0"', validator)
        gitignore = (ROOT / '.gitignore').read_text()
        self.assertIn('.venv-mujoco/', gitignore)

    def test_urdf_joint_partition(self):
        urdf = ET.parse(cfgmod.ASSET_FILE).getroot()
        dofs = {j.get('name') for j in urdf.findall('joint') if j.get('type') != 'fixed'}
        self.assertEqual(dofs, set(cfgmod.LEG_NAMES + cfgmod.WHEEL_NAMES))
        self.assertEqual(len(cfgmod.LEG_NAMES), 12)
        payload = urdf.find("link[@name='top_box']")
        self.assertIsNotNone(payload)
        payload_mass = float(payload.find('./inertial/mass').get('value'))
        self.assertEqual(payload_mass, 90.0)
        self.assertEqual(cfgmod.PAYLOAD_MASS_KG, payload_mass)
        inertia = payload.find('./inertial/inertia')
        self.assertEqual(float(inertia.get('ixx')), 1.5318)
        self.assertEqual(float(inertia.get('iyy')), 2.46555)
        self.assertEqual(float(inertia.get('izz')), 3.1077)
        links = {link.get('name') for link in urdf.findall('link')}
        children = {joint.find('child').get('link') for joint in urdf.findall('joint')}
        self.assertEqual(links - children, {'base'})
        self.assertEqual(len(urdf.findall('transmission')), 16)
        self.assertEqual(len(urdf.findall('gazebo')), 22)

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
            'rsl_rl/modules/dreamwaq_estimator.py',
            'rsl_rl/modules/dreamwaq_actor_critic.py',
            'rsl_rl/algorithms/dreamwaq_ppo.py',
            'rsl_rl/storage/dreamwaq_rollout_storage.py',
            'nezha_stand/runner.py',
            'scripts/check_dreamwaq.py',
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)
        registry_source = (ROOT / 'legged_gym/utils/task_registry.py').read_text()
        self.assertIn("runner_name == 'DreamWaQStandRunner'", registry_source)

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

    def test_optional_wandb_scalar_logging(self):
        pyproject = (ROOT / 'pyproject.toml').read_text()
        train_source = (ROOT / 'scripts/train.py').read_text()
        runner_source = (ROOT / 'nezha_stand/runner.py').read_text()
        self.assertIn('wandb = ["wandb>=0.23,<0.25"]', pyproject)
        self.assertIn('"--wandb"', train_source)
        self.assertIn('"--wandb_project"', train_source)
        self.assertIn('"--wandb_group"', train_source)
        self.assertIn('"--wandb_tags"', train_source)
        self.assertIn('"--wandb_mode"', train_source)
        self.assertIn('external_logger.log(metrics, step=iteration)', runner_source)
        # Curves only: do not silently upload source, checkpoints, or full config.
        self.assertNotIn('save_code=True', train_source)
        self.assertNotIn('sync_tensorboard=True', train_source)
        self.assertNotIn('log_artifact(', train_source + runner_source)
        self.assertIn('disable_code=True', train_source)
        self.assertIn('x_disable_stats=True', train_source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
