from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DogV2Cfg(LeggedRobotCfg):
    pass

class DogV2PPO(LeggedRobotCfgPPO):
    pass


class DogV2Cfg_GaitConditioned(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        observe_gait_commands = True
        observe_clock_inputs = True
        observe_two_prev_actions = True
        num_observations = 75
        num_privileged_obs = 265
        num_obs_hist = 5
        num_history_obs = num_obs_hist * num_observations

    class commands(LeggedRobotCfg.commands):
        resampling_time = 10
        num_commands = 14
        heading_command = True
        exclusive_phase_offset = False
        binary_phases = True

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            body_height = [-0.25, 0.15]
            gait_frequency = [2.0, 4.0]
            gait_phase = [0.0, 1.0]
            gait_offset = [0.0, 1.0]
            gait_bound = [0.0, 1.0]
            gait_duration = [0.5, 0.5]
            footswing_height = [0.03, 0.35]
            body_pitch = [-0.4, 0.4]
            body_roll = [-0.0, 0.0]
            stance_width = [0.15, 0.35]
            stance_length = [0.35, 0.45]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        only_positive_rewards_ji22_style = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            lin_vel_z = -2
            dof_acc = -2.5e-8
            ang_vel_xy = -0.05
            torques = -0.0001
            dof_vel = -1e-4
            collision = -1.0
            dof_pos_limits = -10.0
            feet_slip = -0.04
            feet_clearance_cmd_linear = -10.0
            action_smoothness_1 = -0.0
            action_smoothness_2 = -0.02
            raibert_heuristic = -10.0
            orientation_control = -0.5
            jump = 10.0
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            stand_still = -0.5
            standup = -0.25

        curriculum_rewards = [
            {'reward_name': 'tracking_lin_vel', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.5},
            {'reward_name': 'tracking_ang_vel', 'start_iter': 0, 'end_iter': 1000, 'start_value': 1.0, 'end_value': 0.5},
            {'reward_name': 'jump', 'start_iter': 0, 'end_iter': 1000, 'start_value': 1.0, 'end_value': 4.0},
            {'reward_name': 'raibert_heuristic', 'start_iter': 0, 'end_iter': 1000, 'start_value': 1.0, 'end_value': 2.0},
            {'reward_name': 'dof_acc', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 10.0},
        ]

    class domain_rand(LeggedRobotCfg.domain_rand):
        lag_timesteps = 6
        randomize_lag_timesteps = True
        push_robots = True
        push_interval_s = 9
        max_push_vel_xy = 1.0
        max_push_ang_vel = 0.6
        randomize_friction = True
        friction_range = [0.5, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        added_base_mass_range = [-2., 4.]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]
        randomize_base_com = True
        added_base_com_range = [-0.08, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.5, 1.5]
        torque_multiplier_range = [0.85, 1.15]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.105, 0.105]
        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 3]

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]


class DogV2PPO_GaitConditioned(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        num_obs = 75

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'dog_v2_gait'
