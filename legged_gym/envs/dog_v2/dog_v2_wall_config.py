from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class DogV2WallCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 235
        episode_length_s = 20
        goal_reach_threshold = 0.3
        num_steps_per_env = 24
        test = False
        reach_goal_delay = 0.1
        num_future_goal_obs = 2

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        curriculum = True
        terrain_length = 8.0
        terrain_width = 4.0
        num_rows = 10
        num_cols = 20
        max_init_terrain_level = 5
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 25
        slope_treshold = 0.75
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # Wall-specific
        num_goals = 4
        num_hurdles = 3
        wall_h_min = 0.05
        wall_h_max = 0.5
        wall_positions = [2.0, 4.0, 6.0]
        goal_positions = [1.0, 3.0, 5.0, 7.0]
        plane_env_prob = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 7.0

        class ranges:
            lin_vel_x = [0.4, 1.2]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.0,
            'RL_thigh_joint': 0.0,
            'FR_thigh_joint': 0.0,
            'RR_thigh_joint': 0.0,
            'FL_calf_joint': 0.0,
            'RL_calf_joint': 0.0,
            'FR_calf_joint': 0.0,
            'RR_calf_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 28.}
        damping = {'joint': 0.7}
        action_scale = 0.18
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_v2_description/urdf/backup/dog_v2.urdf'
        name = "dog_v2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_interval_s = 7
        randomize_friction = True
        friction_range = [0.6, 2.0]
        restitution_range = [0.0,0.5]
        randomize_base_mass = True
        added_base_mass_range = [-1., 3.]
        randomize_base_com = True
        added_base_com_range = [-0.1, 0.1]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.25

        class scales:
            termination = -1.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_goal_vel = 2.5
            tracking_yaw = 0.5
            lin_vel_z = -2
            ang_vel_xy = -0.05
            orientation = 0.0
            base_height = 0.0
            torques = -0.0001
            dof_acc = -2.5e-7
            collision = -0.5
            action_rate = -0.1
            stand_still = -0.5
            dof_pos_limits = -5.0
            hip_default = -0.5
            standup = -0.2
            run_still = -0.04
            feet_air_time = 1.0
            turn_contact_number = 0.0
            turn_small_steps = 0

    class normalization(LeggedRobotCfg.normalization):
        pass

    class noise(LeggedRobotCfg.noise):
        pass

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class DogV2WallCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'DreamWaQRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 4
        num_mini_batches = 6
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_obs = 45

    class runner:
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24
        max_iterations = 5000
        save_interval = 100
        experiment_name = 'dog_v2_wall'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
