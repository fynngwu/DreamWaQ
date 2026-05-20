import numpy as np
import os
import shutil
import inspect
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "legged_gym"))
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym import LEGGED_GYM_ROOT_DIR

import torch

def save_run_config(task_name, env_cfg, train_cfg, log_dir):
    if log_dir is None:
        return

    config_dir = os.path.join(log_dir, 'config_snapshot')
    os.makedirs(config_dir, exist_ok=True)

    base_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs', 'base')
    base_files = ['legged_robot.py', 'legged_robot_config.py', 'base_config.py', 'base_task.py']
    for f in base_files:
        src = os.path.join(base_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(config_dir, f))

    env_cfg_src = inspect.getfile(type(env_cfg))
    if env_cfg_src and os.path.exists(env_cfg_src):
        shutil.copy2(env_cfg_src, os.path.join(config_dir, os.path.basename(env_cfg_src)))

    train_cfg_src = inspect.getfile(type(train_cfg))
    if train_cfg_src and os.path.exists(train_cfg_src):
        dst_name = os.path.basename(train_cfg_src)
        if dst_name != os.path.basename(env_cfg_src):
            shutil.copy2(train_cfg_src, os.path.join(config_dir, dst_name))

    task_class = task_registry.get_task_class(task_name)
    task_class_src = inspect.getfile(task_class)
    if task_class_src and os.path.exists(task_class_src):
        shutil.copy2(task_class_src, os.path.join(config_dir, os.path.basename(task_class_src)))

    rl_src_files = [
        os.path.join(LEGGED_GYM_ROOT_DIR, 'rsl_rl', 'rsl_rl', 'runners', 'dreamwaq_runner.py'),
        os.path.join(LEGGED_GYM_ROOT_DIR, 'rsl_rl', 'rsl_rl', 'algorithms', 'ppo_dreamwaq.py'),
        os.path.join(LEGGED_GYM_ROOT_DIR, 'rsl_rl', 'rsl_rl', 'modules', 'actor_critic_dreamwaq.py'),
        os.path.join(LEGGED_GYM_ROOT_DIR, 'rsl_rl', 'rsl_rl', 'modules', 'vae.py'),
        os.path.join(LEGGED_GYM_ROOT_DIR, 'rsl_rl', 'rsl_rl', 'storage', 'rollout_storage_dreamwaq.py'),
    ]
    for src in rl_src_files:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(config_dir, os.path.basename(src)))

    from legged_gym.utils.helpers import class_to_dict
    with open(os.path.join(config_dir, 'config_dump.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== Env Config ===\n")
        f.write(str(class_to_dict(env_cfg)))
        f.write("\n\n=== Train Config ===\n")
        f.write(str(class_to_dict(train_cfg)))

    print(f"Config snapshot saved to: {config_dir}")


def create_experiment_log(task_name, env_cfg, train_cfg, log_dir):
    if log_dir is None:
        return

    md_path = os.path.join(log_dir, 'experiment_log.md')

    from legged_gym.utils.helpers import class_to_dict
    env_cfg_dict = class_to_dict(env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)

    reward_scales = env_cfg_dict.get('rewards', {}).get('scales', {})
    reward_lines = '\n'.join([f"| {k} | {v} |" for k, v in reward_scales.items()])

    command_ranges = env_cfg_dict.get('commands', {}).get('ranges', {})
    command_lines = '\n'.join([f"| {k} | {v} |" for k, v in command_ranges.items()])

    domain_rand_keys = [
        ('randomize_friction', '随机摩擦'),
        ('friction_range', '摩擦范围'),
        ('randomize_restitution', '随机恢复系数'),
        ('push_robots', '随机推力'),
        ('randomize_base_mass', '随机基座质量'),
        ('added_base_mass_range', '基座质量范围'),
        ('randomize_pd_gains', '随机PD增益'),
        ('stiffness_multiplier_range', '刚度倍率范围'),
        ('damping_multiplier_range', '阻尼倍率范围'),
        ('randomize_motor_zero_offset', '随机电机零偏'),
        ('add_cmd_action_latency', '动作延迟'),
        ('range_cmd_action_latency', '动作延迟范围'),
    ]
    domain_rand_lines = '\n'.join([
        f"| {label} | {env_cfg_dict.get('domain_rand', {}).get(k, 'N/A')} |"
        for k, label in domain_rand_keys
    ])

    content = f"""# 实验记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

> 任务: **{task_name}** | 日志: `{log_dir}`

## 观察到的效果



## 问题与分析



## 下一步改进方向



---

## 奖励权重
| 奖励项 | 权重 |
|--------|------|
{reward_lines}

## 训练超参数
| 参数 | 值 |
|------|-----|
| 最大迭代 | {train_cfg_dict.get('runner', {}).get('max_iterations', 'N/A')} |
| 学习率 | {train_cfg_dict.get('algorithm', {}).get('learning_rate', 'N/A')} |
| 保存间隔 | {train_cfg_dict.get('runner', {}).get('save_interval', 'N/A')} |
| batch size (steps/env) | {train_cfg_dict.get('runner', {}).get('num_steps_per_env', 'N/A')} |
| mini batches | {train_cfg_dict.get('algorithm', {}).get('num_mini_batches', 'N/A')} |
| gamma | {train_cfg_dict.get('algorithm', {}).get('gamma', 'N/A')} |
| lambda | {train_cfg_dict.get('algorithm', {}).get('lam', 'N/A')} |
| clip param | {train_cfg_dict.get('algorithm', {}).get('clip_param', 'N/A')} |
| entropy coef | {train_cfg_dict.get('algorithm', {}).get('entropy_coef', 'N/A')} |

## 命令范围
| 命令 | 范围 |
|------|------|
{command_lines}

## 域随机化
| 参数 | 值 |
|------|-----|
{domain_rand_lines}

## 环境配置
| 参数 | 值 |
|------|-----|
| 环境数量 | {env_cfg_dict.get('env', {}).get('num_envs', 'N/A')} |
| 观测维度 | {env_cfg_dict.get('env', {}).get('num_observations', 'N/A')} |
| 历史观测数 | {env_cfg_dict.get('env', {}).get('num_obs_hist', 'N/A')} |
| 动作维度 | {env_cfg_dict.get('env', {}).get('num_actions', 'N/A')} |
| Episode长度 | {env_cfg_dict.get('env', {}).get('episode_length_s', 'N/A')}s |
| 地形类型 | {env_cfg_dict.get('terrain', {}).get('mesh_type', 'N/A')} |
| 地形proportions | {env_cfg_dict.get('terrain', {}).get('terrain_proportions', 'N/A')} |

## PD控制参数
| 参数 | 值 |
|------|-----|
| action_scale | {env_cfg_dict.get('control', {}).get('action_scale', 'N/A')} |
| decimation | {env_cfg_dict.get('control', {}).get('decimation', 'N/A')} |
| stiffness | {env_cfg_dict.get('control', {}).get('stiffness', 'N/A')} |
| damping | {env_cfg_dict.get('control', {}).get('damping', 'N/A')} |

---
*配置副本已保存在 `config_snapshot/` 目录下*
"""

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Experiment log created: {md_path}")


def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    save_run_config(args.task, env_cfg, train_cfg, ppo_runner.log_dir)
    create_experiment_log(args.task, env_cfg, train_cfg, ppo_runner.log_dir)

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
