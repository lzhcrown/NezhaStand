"""DreamWaQ asymmetric training configuration for Nezha standing."""
from pathlib import Path
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_FILE = PROJECT_ROOT / 'assets/nezha/urdf/nezha.urdf'
LEG_NAMES = [f'{leg}_{joint}_joint' for leg in ('FL', 'FR', 'RL', 'RR')
             for joint in ('hip', 'thigh', 'calf')]
WHEEL_NAMES = [f'{leg}_foot_joint' for leg in ('FL', 'FR', 'RL', 'RR')]


class NezhaStandCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_actions = 12
        # omega 3 + gravity 3 + leg q error 12 + leg dq 12 + wheel dq 4 + action 12
        num_one_step_observations = 46
        num_observations = 46
        observation_history_length = 5
        num_observation_history = num_observations * observation_history_length
        # clean actor 46 + true body velocity 3 + height error 1 + foot forces 12 + drift 2
        num_one_step_privileged_obs = 64
        num_privileged_obs = 64
        episode_length_s = 10.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = False
        measure_heights = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.54]
        rot = [0.0, 0.0, 0.0, 1.0]
        joint_jitter = 0.02
        default_joint_angles = {
            f'{leg}_{joint}_joint': value
            for leg in ('FL', 'FR', 'RL', 'RR')
            for joint, value in (
                # Matches Desktop/config.yaml default_dof_pos in FL/FR/RL/RR order.
                ('hip', 0.10 if leg in ('FL', 'RL') else -0.10),
                ('thigh', 0.925), ('calf', -1.85), ('foot', 0.0))}

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_joint': 150.0, 'thigh_joint': 220.0,
                     'calf_joint': 220.0, 'foot_joint': 0.0}
        damping = {'hip_joint': 4.0, 'thigh_joint': 4.0,
                   'calf_joint': 4.0, 'foot_joint': 1.2}
        control_type = 'P'
        action_scale = 0.15
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = str(ASSET_FILE)
        name = 'nezha_stand'
        feet_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        wheel_dof_names = WHEEL_NAMES
        leg_dof_names = LEG_NAMES
        foot_name = 'foot'
        wheel_name = ['foot_joint']
        penalize_contacts_on = ['hip', 'thigh', 'calf']
        terminate_after_contacts_on = ['trunk', 'hip']
        collapse_fixed_joints = True
        self_collisions = 1
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_payload_mass = False
        randomize_com_displacement = False
        randomize_link_mass = False
        randomize_friction = False
        randomize_restitution = False
        randomize_motor_strength = False
        randomize_kp = False
        randomize_kd = False
        randomize_initial_joint_pos = False
        disturbance = False
        push_robots = False
        delay = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        # LZHMine also targets 0.50 m. Unlike a nominal-joint reward, this
        # constrains only body geometry and leaves all leg angles policy-free.
        base_height_target = 0.50
        height_error_scale = 0.10
        height_gate_start = 0.45
        soft_dof_pos_limit = 0.9
        max_contact_force = 450.0
        contact_threshold = 5.0
        termination_cost = -10.0  # one-off penalty, not multiplied by dt
        class scales:
            # Primary four-wheel-on-ground standing objectives.  There is
            # deliberately no default/nominal joint-position reward: the
            # default angles are only the reset pose and PD action origin.
            upright = 3.0
            # LZHMine-style non-saturating squared height error, normalized by
            # 0.10 m so a 0.10 m crouch costs 4 reward units per second.
            height = -4.0
            stationary = 1.0
            support = 2.0

            # Mean squared difference of normalized |torque| between the six
            # leg pairs, compared only between corresponding joint types.
            torque_balance = -4.0

            # Wheel-legged standing terms.  Normalized vertical-load balance
            # prevents a wheel from merely touching with negligible load;
            # foot_slip penalizes translation of a contacting wheel centre.
            foot_force_balance = -0.5
            foot_slip = -0.2
            wheel_speed = -0.05
            drift = -2.0

            # Generic stability, smoothness and hardware-safety regularizers.
            lin_vel_z = -2.0
            ang_vel_xy = -0.2
            collision = -2.0
            action_rate = -0.02
            torques = -2.5e-6
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            dof_pos_limits = -1.0
            feet_contact_forces = -5e-4

    class termination:
        max_tilt_rad = 0.7853981634
        # Nominal URDF geometry places the base near 0.47 m with wheels on the
        # floor. 0.42 m permits settling but rejects the learned crouch.
        min_height = 0.42
        max_displacement = 0.50
        contact_threshold = 5.0

    class evaluation:
        settling_time_s = 1.0
        min_survival = 0.95
        max_roll_pitch_rms_deg = 2.0
        max_height_rmse_m = 0.02
        max_xy_speed_rms_m_s = 0.05
        max_saturation_fraction = 0.01
        min_all_feet_contact_fraction = 0.95
        max_xy_drift_m = 0.10
        max_wheel_rms_rad_s = 0.20
        max_torque_pair_rms_nm = 10.0
        max_foot_load_fraction_rms = 0.10
        max_contact_foot_speed_rms_m_s = 0.05

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 3.0  # +/-0.45 rad residual, additionally clamped to limits
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            ang_vel = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.5

    class viewer(LeggedRobotCfg.viewer):
        pos = [3.0, -3.0, 1.8]
        lookat = [0.0, 0.0, 0.5]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005


class NezhaStandCfgPPO(LeggedRobotCfgPPO):
    seed = 42
    runner_class_name = 'DreamWaQStandRunner'
    class policy:
        init_noise_std = 0.5
        history_length = 5
        latent_dim = 16
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        encoder_hidden_dims = [128]
        decoder_hidden_dims = [64, 128]
        activation = 'elu'
        vae_sigma_min = 0.0
        vae_sigma_max = 5.0
        # Actor observations occupy [0:46]; true body velocity follows in critic obs.
        velocity_target_start = 46
    class algorithm(LeggedRobotCfgPPO.algorithm):
        learning_rate = 1e-3
        vae_learning_rate = 1e-3
        kl_weight = 0.1
        entropy_coef = 0.005
        schedule = 'adaptive'
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'DreamWaQActorCritic'
        algorithm_class_name = 'DreamWaQPPO'
        num_steps_per_env = 48
        max_iterations = 20000
        save_interval = 100
        experiment_name = 'nezha_stand'
        run_name = 'dreamwaq_stand_height_v2'
