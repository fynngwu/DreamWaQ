from legged_gym.envs.base.base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 45
        num_obs_hist = 5
        num_privileged_obs = 235
        num_latent_dims = 16
        num_explicit_dims = 3
        num_history_obs = num_obs_hist * num_observations
        num_actions = 12
        episode_length_s = 20
        env_spacing = 3.
        send_timeouts = True

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.2
        num_commands = 14  # 14维命令: lin_vel_x/y, ang_vel_yaw, body_height, gait_params(5), footswing_height, body_pitch/roll, stance_width/length
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        # --- 课程学习配置 ---
        curriculum_type = "RewardThresholdCurriculum"
        curriculum_seed = 100
        gaitwise_curricula = False  # 是否为 pronk/trot/pace/bound 维护独立课程分布
        exclusive_phase_offset = False
        pacing_offset = False
        binary_phases = True  # 是否将步态相位参数量化为最近0.5
        balance_gait_distribution = False

        class ranges:
            lin_vel_x = [-1.0, 1.2]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]  # min max [m/s]
            ang_vel_yaw = [-2, 2]    # min max [rad/s]
            heading = [-3.14, 3.14]
            body_height = [-0.25, 0.15]  # 期望机身高度偏移 [m]
            gait_frequency = [2.0, 4.0]   # 步频 [Hz]
            gait_phase = [0.0, 1.0]       # 步态相位（脚0与脚2的相位偏移）
            gait_offset = [0.0, 1.0]      # 步态左右偏移
            gait_bound = [0.0, 1.0]       # 步态前后偏移
            gait_duration = [0.5, 0.5]    # 占空比（stance比例）
            footswing_height = [0.03, 0.35]  # 足端摆动高度 [m]
            body_pitch = [-0.4, 0.4]      # 期望机身俯仰角 [rad]
            body_roll = [-0.0, 0.0]       # 期望机身横滚角 [rad]
            stance_width = [0.10, 0.45]   # 期望步宽 [m]
            stance_length = [0.35, 0.45]  # 期望步长 [m]

        # 课程网格的绝对边界（可以比ranges更宽）
        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.25, 0.15]
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_phase = [0.0, 1.0]
        limit_gait_offset = [0.0, 1.0]
        limit_gait_bound = [0.0, 1.0]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll = [-0.0, 0.0]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]

        # 课程网格每个维度的bin数量
        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_roll = 1
        num_bins_body_pitch = 1
        num_bins_stance_width = 1
        num_bins_stance_length = 1

    class curriculum_thresholds:
        # 课程成功判定阈值（与reward scale相乘后作为实际threshold）
        tracking_lin_vel = 0.8
        tracking_ang_vel = 0.5
        tracking_contacts_shaped_force = 0.8
        tracking_contacts_shaped_vel = 0.8

    class init_state:
        pos = [0.0, 0.0, 0.34]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
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

    class control:
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 25.}
        damping = {'joint': 1.}
        action_scale = 0.25
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_v2_description/urdf/dog_v2_2_4.urdf'
        name = "dog_v2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf","base"]
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.55]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        push_robots = True
        push_interval_s = 9
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
        randomize_base_mass = True
        added_base_mass_range = [-2., 4.]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]
        randomize_base_com = True
        added_base_com_range = [-0.08, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.5, 1.5]
        damping_multiplier_range = [0.5, 1.5]
        torque_multiplier_range = [0.85, 1.15]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.135, 0.135] # Offset to add to the motor angles

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 3]

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            action_rate = -0.01
            stand_still = -0.0
            tracking_lin_vel_lat = 0.0
            tracking_lin_vel_long = 0.0
            tracking_contacts = 0.0
            tracking_contacts_shaped = 0.0
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0
            jump = 0.0
            energy = 0.0
            energy_expenditure = 0.0
            survival = 0.0
            dof_pos_limits = 0.0
            feet_contact_forces = 0.0
            feet_slip = 0.0
            feet_clearance_cmd_linear = 0.0
            dof_pos = 0.0
            action_smoothness_1 = 0.0
            action_smoothness_2 = 0.0
            feet_impact_vel = 0.0
            raibert_heuristic = 0.0
            standup = -0.25

        only_positive_rewards = True
        only_positive_rewards_ji22_style = False
        sigma_rew_neg = 5.0
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.38
        max_contact_force = 100. # forces above this value are penalized
        # gait-shaped rewards 参数
        kappa_gait_probs = 0.07  # von Mises 分布平滑系数
        gait_force_sigma = 100.  # 步态接触力高斯核宽度
        gait_vel_sigma = 10.     # 步态足端速度高斯核宽度

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            body_height_cmd = 5.0
            gait_freq_cmd = 0.5
            gait_phase_cmd = 1.0
            footswing_height_cmd = 5.0
            body_pitch_cmd = 2.0
            body_roll_cmd = 1.0
            stance_width_cmd = 5.0
            stance_length_cmd = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            contact_states = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'DreamWaQRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 4
        num_mini_batches = 6 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        num_obs=45
    class runner:
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"
        num_steps_per_env = 24
        max_iterations = 10000
        save_interval = 100
        experiment_name = 'dog_v2'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
