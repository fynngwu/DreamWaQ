from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DogV2Cfg(LeggedRobotCfg):
    pass

class DogV2PPO(LeggedRobotCfgPPO):
    pass


# === gait-conditioned 训练配置（参考 walk-these-ways/CoRL 的设置） ===
class DogV2Cfg_GaitConditioned(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        observe_gait_commands = True
        observe_clock_inputs = True
        observe_two_prev_actions = True
        # 观测维度: vel(6) + projected_gravity(3) + commands(14) + dof_err(12) + dof_vel(12) + actions(12) + last_actions(12) + clock(4) = 75
        num_observations = 75
        num_privileged_obs = 265  # 3lin_vel + 187height + 75obs
        num_obs_hist = 5
        num_history_obs = num_obs_hist * num_observations

    class commands(LeggedRobotCfg.commands):
        resampling_time = 10
        gaitwise_curricula = True
        num_commands = 14
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

        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]
            body_height = [-0.25, 0.15]
            gait_frequency = [2.0, 4.0]
            gait_phase = [0.0, 1.0]
            gait_offset = [0.0, 1.0]
            gait_bound = [0.0, 1.0]
            gait_duration = [0.5, 0.5]
            footswing_height = [0.03, 0.35]
            body_pitch = [-0.4, 0.4]
            body_roll = [-0.0, 0.0]
            stance_width = [0.10, 0.45]
            stance_length = [0.35, 0.45]

    class curriculum_thresholds(LeggedRobotCfg.curriculum_thresholds):
        tracking_ang_vel = 0.7
        tracking_contacts_shaped_vel = 0.90
        tracking_contacts_shaped_force = 0.90

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            torques = -0.00001
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            collision = -5.0
            action_rate = -0.01
            feet_slip = -0.04
            feet_clearance_cmd_linear = -30.0
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1
            raibert_heuristic = -10.0
            orientation_control = -5.0
            jump = 10.0
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            stand_still = -0.5
            standup = -0.25
            dof_pos_limits = -5.0

        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.

    class domain_rand(LeggedRobotCfg.domain_rand):
        lag_timesteps = 6
        randomize_lag_timesteps = True

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.34]


class DogV2PPO_GaitConditioned(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        num_obs = 75  # 必须与 env.num_observations 一致，否则 VAE decoder 维度不匹配

    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = 'dog_v2_gait_conditioned'
