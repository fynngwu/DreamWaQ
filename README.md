# DreamWaQ

Dynamic Robust Adaptive Model-based Whole-body Quadrupedal locomotion — an RL framework for wheel-legged and quadruped robots.

## Deployment

### 1. Clone

```bash
git clone git@github.com:fynngwu/DreamWaQ.git
cd DreamWaQ
git submodule update --init --recursive
```

### 2. Install Dependencies

Python 3.8 required.

```bash
# Isaac Gym (download from https://developer.nvidia.com/isaac-gym)
pip install -e ~/isaacgym/python

# rsl_rl
pip install -e ./rsl_rl

# legged_gym
pip install -e .
```

### 3. Train Dog V2

```bash
python legged_gym/scripts/train.py --task=dog_v2 --headless
```

### 4. Play Policy

```bash
python legged_gym/scripts/play.py --task=dog_v2 --num_envs=50
```

## Supported Tasks

| Task | Description |
|------|-------------|
| `dog_v2` | Custom quadruped robot Dog V2 |
| `m20` | Wheel-legged robot (山猫 M20) |
| `go2_handstand` | Unitree Go2 handstand |
| `go2_legstand` | Unitree Go2 leg stand |

## Submodules

- [dog_v2_description](https://github.com/fynngwu/dog_v2_description) — URDF/MJCF model for Dog V2
- [dogv2_mujoco](https://github.com/fynngwu/dogv2_mujoco) — MuJoCo sim deployment for Dog V2
- [rl_sar-ARES](https://github.com/fynngwu/rl_sar-ARES) — Real robot deployment (ARES framework)

## Acknowledgments

- https://github.com/XinLang2019/Wheel_Legged_Gym
