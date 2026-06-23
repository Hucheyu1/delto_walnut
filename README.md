# Delto Walnut HCY

这是一个基于 Isaac Lab 的强化学习项目，用于训练五指机械手在物理仿真中操控两颗小球，并尽量保持两球围绕指定轴线稳定旋转。项目采用 Isaac Lab 的 `DirectRLEnv` 接口实现环境，使用 RSL-RL 的 PPO 作为默认训练算法。

当前注册的 Gym 任务名是：

```bash
Template-Delto-Walnut-Direct-v0
```

## 项目结构

```text
.
├── README.md
├── scripts/
│   ├── list_envs.py              # 列出已注册任务
│   ├── zero_agent.py             # 零动作调试代理
│   ├── random_agent.py           # 随机动作调试代理
│   └── rsl_rl/
│       ├── train.py              # RSL-RL 训练入口
│       ├── play.py               # checkpoint 播放与策略导出入口
│       └── cli_args.py           # RSL-RL 命令行参数
└── source/delto_walnut_hcy/
    ├── config/extension.toml      # Isaac Lab / Omniverse 扩展配置
    ├── setup.py                   # Python 包安装配置
    └── delto_walnut_hcy/
        └── tasks/direct/delto_walnut_hcy/
            ├── __init__.py
            ├── delto_cfg.py                  # 五指手 USD 与执行器配置
            ├── delto_walnut_hcy_env_cfg.py   # 环境、场景、观测/动作空间配置
            ├── delto_walnut_hcy_env.py       # Direct RL 环境实现
            ├── agents/rsl_rl_ppo_cfg.py      # PPO 默认超参数
            └── robots/dg5f_right.usd         # 机械手资产
```

## 环境概览

- 机器人：`dg5f_right.usd` 五指机械手。
- 操作对象：两个红色球体，默认半径 `0.02 m`，质量 `0.02 kg`。
- 并行环境数：默认 `512`。
- 动作空间：`20` 维，对应 20 个手部关节的增量位置控制。
- 观测空间：`79` 维，包含关节位置/速度、两球位置/速度、上一步动作、旋转轴、旋转中心和球半径。
- 仿真步长：`1 / 120 s`，默认 `decimation = 4`。
- 单回合时长：默认 `10 s`。
- 终止条件：任意球体高度低于 `drop_height_threshold = 0.1`。

奖励主要约束两球的旋转半径、中心对称性、切向速度、绕轴方向、角速度、掉落惩罚、动作平滑、力矩和指间碰撞。环境中启用了课程学习，默认在第 `500` 到 `1000` 个 RSL-RL iteration 之间逐步加入动作质量与碰撞相关约束。

## 依赖

请先安装 Isaac Sim 与 Isaac Lab

建议环境：
- Python `3.11`
- Isaac Lab  `2.3.0`
- Isaac Sim  `5.1.0`


## 安装

在本仓库根目录执行：

```bash
python -m pip install -e source/delto_walnut_hcy
```

安装后可以检查任务是否注册成功：

```bash
python scripts/list_envs.py
```

## 快速运行
### 训练
```bash
python scripts/rsl_rl/train.py --task Template-Delto-Walnut-Direct-v0 --headless
```

训练日志：
```bash
tensorboard --logdir /root/gpufree-data/lab_lecture/delto_walnut_hcy/logs/rsl_rl/delto_walnut
```

常用参数：
```bash
--num_envs 128              # 覆盖并行环境数量
--max_iterations 1000       # 覆盖训练迭代数
--seed 42                   # 指定随机种子
--video                     # 训练时录制视频
--video_length 200          # 视频步数
--video_interval 2000       # 视频录制间隔
--resume                    # 从 checkpoint 恢复
--load_run <run_dir>        # 指定恢复的 run 目录
--checkpoint <model.pt>     # 指定恢复的 checkpoint
```

### 播放与导出策略
```bash
python scripts/rsl_rl/play.py --task Template-Delto-Walnut-Direct-v0 --num_envs 1 
```

如需录制播放视频：

```bash
python scripts/rsl_rl/play.py --task Template-Delto-Walnut-Direct-v0 --num_envs 1 --video --video_length 500 --headless
```

### 蒸馏训练
teacher-student 蒸馏，用已有 79 维 teacher checkpoint
```bash
python scripts/rsl_rl/train.py \
  --task Template-Delto-Walnut-Direct-v2 \
  --agent rsl_rl_distill_cfg_entry_point \
  --headless \
  --load_run 2026-06-12_23-13-39 \
  --checkpoint model_1999.pt \
  --decimation 2 \
  --run_name distill_60hz
```

从头开始训练reduced actor，不对称 PPO：actor 看 53 维，critic 看 79 维
```bash
python scripts/rsl_rl/train.py \
  --task Template-Delto-Walnut-Direct-v2 \
  --decimation 2 \
  --headless
```

### 播放与导出策略
测试蒸馏出来的 student 模型，这个要加 --agent
```bash
python scripts/rsl_rl/play.py \
  --task Template-Delto-Walnut-Direct-v2 \
  --agent rsl_rl_distill_cfg_entry_point \
  --num_envs 1 \
  --decimation 2 \
  --checkpoint /root/gpufree-data/lab_lecture/delto_walnut_hcy/logs/rsl_rl/delto_walnut/你的distill训练目录/model_1999.pt \
  --headless
```
测试 v2 用 PPO 重新训练出来的模型
```bash
python scripts/rsl_rl/play.py \
  --task Template-Delto-Walnut-Direct-v2 \
  --num_envs 1 \
  --decimation 2 \
  --video \
  --video_length 500 \
  --headless \
  --checkpoint /root/gpufree-data/lab_lecture/delto_walnut_hcy/logs/rsl_rl/delto_walnut/你的v2训练目录/model_1999.pt
```

### 监督学习训练
```bash
python3 scripts/train_supervised_policy.py \
    --obs_mode full \
    --target_mode actions \
    --epochs 800 \
    --batch_size 512 \
    --hidden_dims 512 256 128 \
    --dropout 0.1 \
    --weight_decay 1e-4 \
    --val_ratio 0.2 \
    --use_augmentation \
    --use_history_stack \
    --history_len 10
``` 
--target_mode actions: 表示学习 当前观测 + 上一帧动作 -> 当前记录动作
--target_mode next_actions: 预测“下一帧动作”
--target_mode next_joint_delta: 预测“下一帧关节角相对当前关节角的增量”

### 监督学习测试
```bash
python scripts/test_supervised_policy_on_csv.py \
    --csv /home/amlrobotics/hcy_ws/delto_walnut_hcy/data/replay_data_0615_30HZ_1/replay_data_0615_30HZ_1.csv \
    --policy /home/amlrobotics/hcy_ws/delto_walnut_hcy/logs/supervised_action_policy/supervised_action_policy_jit.pt \
    --num_samples 100 \
    --save_result /home/amlrobotics/hcy_ws/delto_walnut_hcy/logs/supervised_action_policy/test_result.csv

python scripts/test_supervised_policy.py --obs_mode no_vision --target_mode action_delta --use_history_stack --history_len 10
```    

## 代码格式化

项目包含 Ruff 与 pre-commit 配置。安装并运行：
```bash
pip install pre-commit
pre-commit run --all-files
```

## VS Code 配置
仓库保留了 Isaac Lab 模板中的 VS Code 环境配置工具。可以在 VS Code 中运行任务 `setup_python_env`，按提示填写 Isaac Sim 安装路径。生成的 `.vscode/.python.env` 会帮助 Pylance 索引 Isaac Sim、Omniverse 和 Isaac Lab 的扩展路径。

