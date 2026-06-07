import numpy as np
from collections import defaultdict

from isaacgym import terrain_utils

from legged_gym.utils.parkour_hurdle_terrain import convert_heightfield_to_trimesh


class WallTerrain:
    def __init__(self, cfg, num_robots):
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border

        num_hurdles = getattr(cfg, 'num_hurdles', 3)
        num_goals = getattr(cfg, 'num_goals', 4)

        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.int64)
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, num_goals, 3), dtype=np.float32)
        self.hurdles = np.zeros((cfg.num_rows, cfg.num_cols, num_hurdles, 3), dtype=np.float32)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.name2cols = defaultdict(set)
        self.cols2id = []

        self._build_map()
        self.heightsamples = self.height_field_raw

        if cfg.mesh_type != "trimesh":
            raise ValueError("Wall terrain only supports trimesh")
        self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(
            self.height_field_raw,
            cfg.horizontal_scale,
            cfg.vertical_scale,
            cfg.slope_treshold,
        )

    def _build_map(self):
        cfg = self.cfg
        difficulty_den = max(cfg.num_rows - 1, 1)
        num_plane_cols = int(cfg.num_cols * getattr(cfg, 'plane_env_prob', 0.0))
        num_hurdles = getattr(cfg, 'num_hurdles', 3)
        num_goals = getattr(cfg, 'num_goals', 4)

        # Wall positions along the env length (meters from start)
        wall_x_m = getattr(cfg, 'wall_positions', [2.0, 4.0, 6.0])
        goal_x_m = getattr(cfg, 'goal_positions', [1.0, 3.0, 5.0, 7.0])

        for col in range(cfg.num_cols):
            is_plane = col < num_plane_cols
            for row in range(cfg.num_rows):
                difficulty = row / difficulty_den
                terrain = terrain_utils.SubTerrain(
                    "wall_terrain",
                    width=self.length_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=cfg.vertical_scale,
                    horizontal_scale=cfg.horizontal_scale,
                )
                if is_plane:
                    self._make_plane(terrain, num_goals, num_hurdles, goal_x_m)
                else:
                    self._make_wall_track(terrain, difficulty, num_goals, num_hurdles,
                                          wall_x_m, goal_x_m)
                self._add_terrain_to_map(terrain, row, col)
            type_name = "plane" if is_plane else "wall"
            self.name2cols[type_name].add(col)
            self.cols2id.append(0 if is_plane else 16)

    def _make_plane(self, terrain, num_goals, num_hurdles, goal_x_m):
        cfg = self.cfg
        h = terrain.horizontal_scale
        goals = np.zeros((num_goals, 2), dtype=np.float32)
        hurdles = np.zeros((num_hurdles, 3), dtype=np.float32)

        mid_y = terrain.length // 2
        mid_y_m = mid_y * h

        terrain.height_field_raw[:, :] = 0

        for i, gx in enumerate(goal_x_m):
            goals[i] = [gx, mid_y_m]

        terrain.goals = goals
        terrain.hurdles = hurdles
        terrain.idx = 0

    def _make_wall_track(self, terrain, difficulty, num_goals, num_hurdles,
                         wall_x_m, goal_x_m):
        cfg = self.cfg
        h = terrain.horizontal_scale
        v = terrain.vertical_scale

        goals = np.zeros((num_goals, 2), dtype=np.float32)
        hurdles = np.zeros((num_hurdles, 3), dtype=np.float32)

        mid_y = terrain.length // 2
        mid_y_m = mid_y * h

        # Flat ground
        terrain.height_field_raw[:, :] = 0

        # Wall dimensions (with randomization)
        wall_height_base = getattr(cfg, 'wall_h_min', 0.15) + \
            (getattr(cfg, 'wall_h_max', 0.45) - getattr(cfg, 'wall_h_min', 0.15)) * difficulty

        for i, wx in enumerate(wall_x_m):
            # Randomise height ±20% and thickness 0.03-0.08m per-wall
            h_rand = 1.0 + np.random.uniform(-0.1, 0.1)
            wall_height_m = wall_height_base * h_rand
            wall_height_px = max(1, round(wall_height_m / v))
            wall_thick_m = np.random.uniform(0.05, 0.16)
            wall_width_px = max(1, round(wall_thick_m / h))
            wall_len_px = round(3.0 / h)
            wx_px = round(wx / h)
            x0 = max(wx_px - wall_width_px // 2, 0)
            x1 = min(x0 + wall_width_px, terrain.width)
            y0 = max(mid_y - wall_len_px // 2, 0)
            y1 = min(y0 + wall_len_px, terrain.length)
            terrain.height_field_raw[x0:x1, y0:y1] = wall_height_px
            hurdles[i] = [wx, mid_y_m, wall_height_m]

        for i, gx in enumerate(goal_x_m):
            goals[i] = [gx, mid_y_m]

        terrain.goals = goals
        terrain.hurdles = hurdles
        terrain.idx = 16

    def _add_terrain_to_map(self, terrain, row, col):
        cfg = self.cfg
        start_x = self.border + row * self.length_per_env_pixels
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = row * self.env_length
        env_origin_y = (col + 0.5) * self.env_width
        self.env_origins[row, col] = [env_origin_x, env_origin_y, 0.0]
        self.terrain_type[row, col] = terrain.idx

        self.goals[row, col, :, :2] = terrain.goals + [row * self.env_length, col * self.env_width]
        self.goals[row, col, :, 2] = 0.0

        if len(terrain.hurdles) > 0:
            self.hurdles[row, col, :, :2] = terrain.hurdles[:, :2] + [row * self.env_length, col * self.env_width]
            self.hurdles[row, col, :, 2] = terrain.hurdles[:, 2]
