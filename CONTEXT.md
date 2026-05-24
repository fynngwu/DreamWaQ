# CONTEXT.md — DreamWaQ / legged_gym

## 术语表

### 机器人
- **DogV2Robot** — 四足机器人环境类，继承自 `LeggedRobot`，是训练的主要目标机器人
- **M20** — 轮式四足机器人（独立任务，不涉及本次迁移）
- **GO2** — Unitree Go2 站立任务（独立任务，不涉及本次迁移）

### 命令系统 (Commands)
- **commands tensor** — 形状 `[num_envs, num_commands]` 的张量，存储当前每个环境的期望命令
- **14维命令向量** — 索引布局：
  - `[0]` = `x_vel` — 前向/后向线速度 (m/s)
  - `[1]` = `y_vel` — 侧向线速度 (m/s)
  - `[2]` = `yaw_vel` — 偏航角速度 (rad/s)
  - `[3]` = `body_height` — 期望机身高度偏移 (m)
  - `[4]` = `gait_frequency` — 步频 (Hz)
  - `[5]` = `gait_phase` — 步态相位（脚 0 与脚 2 之间的相位偏移）
  - `[6]` = `gait_offset` — 步态左右偏移（左/右足对之间的偏移）
  - `[7]` = `gait_bound` — 步态前后偏移（前/后足对之间的偏移）
  - `[8]` = `gait_duration` — 占空比（stance 在步态周期中的比例）
  - `[9]` = `footswing_height` — 足端摆动轨迹最高点高度 (m)
  - `[10]` = `body_pitch` — 期望机身俯仰角 (rad)
  - `[11]` = `body_roll` — 期望机身横滚角 (rad)
  - `[12]` = `stance_width` — 期望步宽（足间距，侧向）(m)
  - `[13]` = `stance_length` — 期望步长（足间距，前后）(m)
- **command_ranges** — 命令采样的初始范围字典，由 `cfg.commands.ranges` 转换而来
- **limit ranges** — 课程网格的绝对边界，可以比 `command_ranges` 更宽
- **resampling_time** — 命令重采样间隔（秒），默认 10s

### 步态系统 (Gait System)
- **clock_inputs** — 每只脚的时钟信号，形如 `[sin(2π·idx), sin(4π·idx), sin(π·idx)]`，共 4 足 × 3 = 12 维
- **foot_indices** — 每只脚在步态周期中的相位索引 [0, 1) 范围
- **desired_contact_states** — 期望接触状态，由 von Mises 分布 CDF 生成 [0,1] 平滑值，用于 gait-shaped rewards
- **gait-wise 类别** — `['pronk', 'trot', 'pace', 'bound']` 四种步态类型，由 (phase, offset, bound) 参数决定
- **binary_phases** — 是否将步态相位参数量化为最近 0.5（开关式相位关系）

### 课程学习 (Curriculum)
- **RewardThresholdCurriculum** — 基于奖励阈值的课程系统，在离散化的多维网格上维护权重分布
- **网格 (grid)** — 所有命令维度的笛卡尔积，每个维度由 `num_bins_*` 决定离散化精度
- **权重 (weights)** — 每个网格单元的选择概率权重，成功时增加 0.2，相邻单元也扩散增加
- **local_range** — 成功时权重扩散到相邻单元的半径
- **gaitwise_curricula** — 是否为每种步态类型维护独立的课程分布
- **curriculum_thresholds** — 判定"成功"的奖励阈值，使用四个指标 AND 组合

### 奖励系统 (Rewards)
- **only_positive_rewards** — 总奖励 clip 到非负值（默认方式，`total = max(sum(rewards), 0)`）
- **ji22 风格正奖励处理** — `total = pos * exp(neg / sigma_rew_neg)`，用指数衰减柔和处理负奖励（当前未使用）
- **gait-shaped rewards** — 依赖于期望接触状态的奖励项（tracking_contacts_shaped_force/vel）
- **tracking_sigma** — 速度跟踪奖励的指数衰减系数（默认 0.25）

### 观测系统 (Observations)
- **obs_buf** — 策略观测张量，形状 `[num_envs, num_observations]`
- **privileged_obs_buf** — 特权观测张量，形状 `[num_envs, num_privileged_obs]`，包含 `[base_lin_vel(3), heights(187), obs_buf(75)]`
- **num_observations** — 观测维度（当前 75），必须与 obs_buf 实际拼接维度严格一致
- **num_privileged_obs** — 特权观测维度（当前 265），等于 3 + 187 + num_observations
- **num_history_obs** — 历史观测维度，等于 `num_obs_hist * num_observations`（当前 375）
- **observe_gait_commands** — 是否将完整 14 维命令加入观测
- **observe_clock_inputs** — 是否将时钟信号加入观测（当前仅 4 维 sin(2π·idx)）

### DreamWaQ 特定架构
- **VAE (CENet)** — 条件编码网络，从观测历史中提取隐变量，包含速度估计和潜在编码分支
- **num_decode_dims** — VAE 解码器输出维度，等于 `num_observations`
- **recon_loss 目标** — `decode_target = critic_obs[:, -num_obs:]`，即 privileged_obs 的最后 num_obs 维（obs_buf 全部 75 维），但 MSE 只计算在 DOF 切片 `[23:47]`（dof_err 12 + dof_vel 12 = 24 维）上，不重建命令、时钟信号等其他内容
- **obs_hist_buf** — VAE 编码器输入，形状 `[num_envs, num_history_obs]`，通过 FIFO 滑动窗口维护
- **ActorCriticDreamWaQ** — 策略-价值网络，actor 输入为 `[code(vel+latent), observation]`
