import torch
import numpy as np

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import torch_rand_float, quat_rotate_inverse

from legged_gym.envs.dog_v2.dog_v2 import DogV2Robot

from legged_gym.utils.wall_terrain import WallTerrain


def euler_from_quaternion(quat):
    # Isaac Gym root state quaternion: [x, y, z, w]
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch = torch.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)
    return roll, pitch, yaw


class DogV2WallRobot(DogV2Robot):
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_steps_per_env = self.cfg.env.num_steps_per_env

    def update_reward_curriculum(self, force_update=False):
        curriculum = getattr(self.cfg.rewards, 'curriculum_rewards', None)
        if curriculum is None and not force_update:
            return
        if curriculum is None:
            curriculum = []
        for entry in curriculum:
            name = entry['reward_name']
            if name not in self.reward_scales:
                continue
            start_iter = entry['start_iter']
            end_iter = entry['end_iter']
            start_val = entry['start_value']
            end_val = entry['end_value']
            if self.common_step_counter is None:
                progress = 0.0
            else:
                current_iter = self.common_step_counter // (self.max_episode_length * self.num_steps_per_env)
                if end_iter > start_iter:
                    progress = min(max((current_iter - start_iter) / (end_iter - start_iter), 0.0), 1.0)
                else:
                    progress = 1.0 if current_iter >= start_iter else 0.0
            self.reward_scales[name] = start_val + (end_val - start_val) * progress

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params,
        )
        self.terrain = WallTerrain(self.cfg.terrain, self.num_envs)
        self._create_trimesh()
        self._create_envs()

    def _init_buffers(self):
        super()._init_buffers()

        # Load terrain goals/hurdles
        self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
        self.terrain_hurdles = torch.from_numpy(self.terrain.hurdles).to(self.device).to(torch.float)
        num_goals = self.terrain_goals.shape[2]

        # Pad env_goals with last-goal extension (for future goal obs)
        num_future = getattr(self.cfg.env, 'num_future_goal_obs', 2)
        total_goals = num_goals + num_future
        self.env_goals = torch.zeros(self.num_envs, total_goals, 3, device=self.device, dtype=torch.float)
        self.env_hurdles = torch.zeros(self.num_envs, self.terrain_hurdles.shape[2], 3, device=self.device, dtype=torch.float)
        self.env_hurdles[:] = self.terrain_hurdles[self.terrain_levels, self.terrain_types]
        self._refresh_env_goals()

        # Near-hurdle mask (|rel_x| < 0.2)
        self.near_hurdle_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Goal tracking state
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.next_target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.next_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reach_goal_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reached_goal_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Roll/pitch/yaw buffers (computed each step from quaternion)
        self.roll = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.pitch = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # Cache cur_goals / next_goals as indexed views
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        # last_torques (needed by train.py reset even if delta_torques reward not used)
        self.last_torques = torch.zeros_like(self.torques)

        # Success flag
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.term_timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.term_roll_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.term_pitch_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._update_goals()

    def _refresh_env_goals(self):
        num_goals = self.terrain_goals.shape[2]
        num_future = getattr(self.cfg.env, 'num_future_goal_obs', 2)
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        self.env_goals[:] = torch.cat((temp, last_col.repeat(1, num_future, 1)), dim=1)

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(
            1, (self.cur_goal_idx[:, None, None] + future).expand(-1, -1, self.env_goals.shape[-1])
        ).squeeze(1)

    def _update_goals(self):
        # 1. Timer-based goal advance (must stay near goal for delay steps)
        delay_steps = getattr(self.cfg.env, 'reach_goal_delay', 0.1) / self.dt
        next_flag = self.reach_goal_timer > delay_steps
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0.0

        # 2. Check if within threshold → increment timer
        self.reached_goal_ids = torch.norm(
            self.root_states[:, :2] - self.cur_goals[:, :2], dim=1
        ) < self.cfg.env.goal_reach_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        # 3. Update relative targets
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # euler angles (matched to extreme-parkour)
        self.roll[:], self.pitch[:], self.yaw[:] = euler_from_quaternion(self.base_quat)

        self._update_goals()
        self._post_physics_step_callback()

        self.check_termination()
        self.near_hurdle_mask = self._compute_near_hurdle_mask()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # Update goal lookups AFTER reset_idx (matched to extreme-parkour)
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # Yaw command toward target (matched to extreme-parkour heading style)
        yaw_error = self.target_yaw - self.yaw
        yaw_cmd = torch.clamp(yaw_error * 2.0, -1.0, 1.0)
        yaw_cmd = torch.where(torch.abs(yaw_cmd) < 0.1, torch.zeros_like(yaw_cmd), yaw_cmd)
        # Only steer when speed command is active
        self.commands[:, 2] = torch.where(
            torch.abs(self.commands[:, 0]) >= 0.1,
            yaw_cmd,
            torch.zeros_like(yaw_cmd),
        )

        self.standup_clamp_factor = torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.env.test and self.viewer and getattr(self, 'enable_viewer_sync', True):
            self._draw_debug_viz()

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Roll/pitch (logged only)
        self.term_roll_buf = torch.abs(self.roll) > 1.5
        self.term_pitch_buf = torch.abs(self.pitch) > 1.5

        # Timeout
        self.term_timeout_buf = self.episode_length_buf > self.max_episode_length
        self.time_out_buf = self.term_timeout_buf.clone()

        # Success (matched to extreme-parkour: reached last real goal)
        self.success_buf = self.cur_goal_idx >= self.cfg.terrain.num_goals
        self.time_out_buf |= self.success_buf

        self.reset_buf |= self.time_out_buf

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        # 10% chance: zero speed
        zero_mask = torch.rand(len(env_ids), device=self.device) < 0.10
        self.commands[env_ids[zero_mask], 0] = 0.0
        self.commands[env_ids, 1:4] = 0.0

    def _get_env_origins(self):
        super()._get_env_origins()
        num_plane_cols = int(self.cfg.terrain.num_cols *
                              getattr(self.cfg.terrain, 'plane_env_prob', 0.0))
        self.is_plane = self.terrain_types < num_plane_cols

        # Test/play mode: randomise across last 3 difficulty levels
        if self.cfg.env.test:
            lo = max(0, self.max_terrain_level - 3)
            self.terrain_levels[:] = torch.randint(lo, self.max_terrain_level, (self.num_envs,), device=self.device)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return

        wall_ids = env_ids[~self.is_plane[env_ids]]
        if len(wall_ids) == 0:
            return

        goals_reached = self.cur_goal_idx[wall_ids]
        move_up = goals_reached >= 3
        move_down = goals_reached <= 1

        self.terrain_levels[wall_ids] += 1 * move_up - 1 * move_down
        # Cap at max, then randomise within high-difficulty band (levels 4-9)
        over_max = self.terrain_levels[wall_ids] >= self.max_terrain_level
        self.terrain_levels[wall_ids] = torch.where(
            over_max,
            torch.randint(4, self.max_terrain_level, wall_ids.shape, device=self.device),
            torch.clip(self.terrain_levels[wall_ids], 0),
        )

        self.env_origins[wall_ids] = self.terrain_origins[self.terrain_levels[wall_ids], self.terrain_types[wall_ids]]
        self.env_hurdles[wall_ids] = self.terrain_hurdles[self.terrain_levels[wall_ids], self.terrain_types[wall_ids]]

        # Refresh env_goals after terrain level change
        self._refresh_env_goals()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        if self.cfg.terrain.curriculum and not self.cfg.env.test:
            self._update_terrain_curriculum(env_ids)
        elif self.cfg.env.test:
            # Play mode: randomise across last 3 difficulty levels
            lo = max(0, self.max_terrain_level - 3)
            self.terrain_levels[env_ids] = torch.randint(lo, self.max_terrain_level, env_ids.shape, device=self.device)
            self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            self.env_hurdles[env_ids] = self.terrain_hurdles[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            self._refresh_env_goals()

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_pd_gains:
            self.p_gains_multiplier[env_ids, :] = torch_rand_float(
                self.cfg.domain_rand.stiffness_multiplier_range[0],
                self.cfg.domain_rand.stiffness_multiplier_range[1],
                (len(env_ids), self.num_actions), device=self.device)
            self.d_gains_multiplier[env_ids, :] = torch_rand_float(
                self.cfg.domain_rand.damping_multiplier_range[0],
                self.cfg.domain_rand.damping_multiplier_range[1],
                (len(env_ids), self.num_actions), device=self.device)
            self.torques_multiplier[env_ids, :] = torch_rand_float(
                self.cfg.domain_rand.torque_multiplier_range[0],
                self.cfg.domain_rand.torque_multiplier_range[1],
                (len(env_ids), self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_motor_zero_offset:
            self.motor_zero_offsets[env_ids, :] = torch_rand_float(
                self.cfg.domain_rand.motor_zero_offset_range[0],
                self.cfg.domain_rand.motor_zero_offset_range[1],
                (len(env_ids), self.num_actions), device=self.device)

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self._reset_latency_buffer(env_ids)

        # Reset goal state (matched to extreme-parkour)
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0.0

        # Refresh goal caches BEFORE _update_goals so targets are correct
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.terrain.mesh_type == "trimesh":
            wall_ids = env_ids[~self.is_plane[env_ids]]
            if len(wall_ids) > 0:
                self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels[wall_ids].float())
            else:
                self.extras["episode"]["terrain_level"] = 0.0
        if len(env_ids) > 0:
            self.extras["episode"]["term_roll"] = torch.mean(self.term_roll_buf[env_ids].float())
            self.extras["episode"]["term_pitch"] = torch.mean(self.term_pitch_buf[env_ids].float())
            self.extras["episode"]["term_timeout"] = torch.mean(self.term_timeout_buf[env_ids].float())
            self.extras["episode"]["term_success"] = torch.mean(self.success_buf[env_ids].float())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids], self.root_states[env_ids, 10:13])

        self._update_goals()

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device)
        # Random initial yaw
        yaw_init = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3] = 0.0  # qx
        self.root_states[env_ids, 4] = 0.0  # qy
        self.root_states[env_ids, 5] = torch.sin(yaw_init / 2)  # qz
        self.root_states[env_ids, 6] = torch.cos(yaw_init / 2)  # qw
        # 10% chance: random fallen pose (sideways or upside down)
        fall_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        fall_env_ids = env_ids[fall_mask]
        if len(fall_env_ids) > 0:
            angles = torch_rand_float(np.pi/2, np.pi, (len(fall_env_ids), 1), device=self.device).squeeze(1)
            neg_mask = torch.rand(len(fall_env_ids), device=self.device) < 0.5
            angles[neg_mask] = -angles[neg_mask]
            half = angles / 2
            self.root_states[fall_env_ids, 3] = torch.sin(half)
            self.root_states[fall_env_ids, 4] = 0.
            self.root_states[fall_env_ids, 5] = 0.
            self.root_states[fall_env_ids, 6] = torch.cos(half)
            self.root_states[fall_env_ids, 2] = self.env_origins[fall_env_ids, 2] + 0.23
        # Random initial velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _draw_debug_viz(self):
        if not self.viewer or not getattr(self, 'enable_viewer_sync', True):
            return
        self.gym.clear_lines(self.viewer)

        all_goal_sphere = gymutil.WireframeSphereGeometry(0.04, 8, 6, None, color=(0.5, 0.5, 0.5))
        cur_goal_sphere = gymutil.WireframeSphereGeometry(0.12, 32, 24, None, color=(0.0, 0.25, 1.0))
        next_goal_sphere = gymutil.WireframeSphereGeometry(0.08, 24, 16, None, color=(0.0, 1.0, 0.35))

        for goals in self.terrain.goals.reshape(-1, self.cfg.terrain.num_goals, 3):
            for goal in goals:
                if np.linalg.norm(goal[:2]) > 0.01:
                    pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], 0.3), r=None)
                    gymutil.draw_lines(all_goal_sphere, self.gym, self.viewer, self.envs[0], pose)

        for env_id in range(min(3, self.num_envs)):
            cur_idx = int(self.cur_goal_idx[env_id].detach().cpu().item())
            goals = self.env_goals[env_id].detach().cpu().numpy()
            for i, g in enumerate(goals):
                if np.linalg.norm(g[:2]) <= 0.01:
                    continue
                if i == cur_idx:
                    sphere = cur_goal_sphere
                elif i == cur_idx + 1:
                    sphere = next_goal_sphere
                else:
                    continue
                pose = gymapi.Transform(gymapi.Vec3(g[0], g[1], 0.3), r=None)
                gymutil.draw_lines(sphere, self.gym, self.viewer, self.envs[0], pose)

    # -------- Hurdle proximity --------

    def _compute_near_hurdle_mask(self):
        """|rel_x| < 0.2 for any hurdle (y-axis unrestricted)."""
        rel = self.env_hurdles[:, :, 0] - self.root_states[:, None, 0]  # (N, num_hurdles)
        near_x = torch.abs(rel) < 0.2
        return near_x.any(dim=1)  # (N,)

    # -------- Reward functions --------

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel_global = self.root_states[:, 7:9]
        forward_vel = torch.sum(target_vec_norm * cur_vel_global, dim=-1)

        cmd_gate = torch.abs(self.commands[:, 0]) > 0.05

        # Normal tracking reward when moving
        rew = torch.minimum(forward_vel, self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        rew = rew * cmd_gate

        # Still reward: zero cmd → reward being motionless (body-frame lin_vel + ang_vel)
        lin_still = torch.exp(-torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1) / self.cfg.rewards.tracking_sigma)
        ang_still = torch.exp(-torch.square(self.base_ang_vel[:, 2]) / self.cfg.rewards.tracking_sigma)
        still_rew = (0.5 * lin_still + 0.5 * ang_still) * (~cmd_gate)

        return (rew + still_rew) * self.standup_clamp_factor

    def _reward_tracking_yaw(self):
        return torch.exp(-torch.abs(self.target_yaw - self.yaw)) * self.standup_clamp_factor * (torch.abs(self.commands[:, 0]) > 0.05)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2]) * (1.0 - 0.5 * self.near_hurdle_mask.float())

    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        return rew * (1.0 - 0.5 * self.near_hurdle_mask.float())

    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
        return rew * (1.0 - 0.5 * self.near_hurdle_mask.float())

    def _reward_collision(self):
        rew = torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        return rew * (1.0 - self.near_hurdle_mask.float())

    def _reward_action_rate(self):
        rew = torch.norm(self.last_actions - self.actions, dim=1)
        return rew * (1.0 - 0.5 * self.near_hurdle_mask.float())

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_hip_default(self):
        hip_err = torch.sum((self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]) ** 2, dim=1)
        return hip_err * (1.0 - self.near_hurdle_mask.float())

    def _reward_run_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        gate = (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.25)
        return torch.sum(torch.abs(dof_err), dim=1) * gate

    def _reward_stand_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.abs(self.commands[:, 0]) < 0.1)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        self.feet_air_time *= ~contact_filt
        return rew_airTime * self.standup_clamp_factor 

