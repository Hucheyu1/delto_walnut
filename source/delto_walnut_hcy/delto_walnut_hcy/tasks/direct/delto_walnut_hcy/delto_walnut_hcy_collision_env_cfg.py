# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .delto_cfg import TESOLLO_CFG


@configclass
class DeltoWalnutCollisionEnvCfg(DirectRLEnvCfg):
    # 环境配置参数（已经集成到命令行参数中）
    decimation = 8  # 环境决策频率，控制仿真步长
    if decimation == 2:
        action_scale = 0.5
        episode_length_s = 10.0  # 每个episode的持续时间（秒）
    if decimation == 4:
        action_scale = 0.5  # 1
        episode_length_s = 10.0  # 每个episode的持续时间（秒）
    if decimation == 8:
        action_scale = 0.5  # 低控制频率下保留足够动作幅度，否则小球难以起转
        episode_length_s = 20.0  # 每个episode的持续时间（秒）
    # 观察空间和动作空间配置
    action_space = 20  # 动作空间维度
    observation_space = 79  # 观察空间维度
    state_space = 0  # 状态空间维度（0表示不使用状态空间）
    drop_height_threshold: float = 0.1  # 物体掉落高度阈值，用于判断是否掉落

    # 超参数在_get_curriculum_weights调节
    w_radius = 1.0
    w_symmetry = 1.0
    w_omega = 1.0
    w_axis = 1.0
    w_dropped = 1.0
    w_tangential = 1.0
    w_joint_smooth = 1.0
    w_torque = 1.0
    w_collision = 1.0

    # ---------------- Kernel reward parameters ----------------
    radius_kernel_sigma = 0.01  # 半径误差容忍，约 5 mm
    sym_center_kernel_sigma = 0.015  # 中心对称误差容忍，约 8 mm
    sym_axis_kernel_sigma = 0.015  # 轴向误差容忍，约 8 mm
    sym_perp_kernel_sigma = 0.015  # 垂直镜像误差容忍，约 8 mm
    axis_kernel_sigma = 0.5
    tan_kernel_sigma = 0.3  # tangential_err 范围大约 [0, 1]
    omega_kernel_sigma = 1.0  # 角速度饱和尺度
    speed_kernel_sigma = 0.05  # 速度门控尺度，单位大致 m/s

    # curriculum learning
    enable_curriculum = True
    # RSL-RL 中每个 iteration 采样 num_steps_per_env 步
    # 你之前设置的是 num_steps_per_env = 16
    curriculum_steps_per_iter = 16
    # 动作质量优化从第多少个 iteration 开始
    quality_start_iter = 500
    # 动作质量优化到第多少个 iteration 完全启用
    quality_end_iter = 1000
    # 对齐 rl_games 中 reward_shaper.scale_value = 0.1
    reward_scale = 0.1

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=1,
        physx=PhysxCfg(
            gpu_collision_stack_size=2**27,  # 约 134 MB，覆盖日志里的 116 MB
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.3,  # 静摩擦系数
            dynamic_friction=0.3,  # 动摩擦系数
        ),
    )

    # ============================
    # 视频录制视角配置
    # ============================
    viewer: ViewerCfg = ViewerCfg(
        origin_type="env",
        env_index=0,
        # 相机位置：沿双球旋转轴 [-0.707, 0, -0.707] 的反方向观察
        eye=(0.476, -0.006, 0.574),
        # 看向两个球和手指接触中心
        lookat=(0.158, -0.006, 0.256),
        resolution=(1920, 1080),
    )

    # 机器人配置
    robot_cfg: ArticulationCfg = TESOLLO_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # 机器人配置

    # 手部关节名称列表
    hand_joint_names = [
        "rj_dg_1_1",
        "rj_dg_1_2",
        "rj_dg_1_3",
        "rj_dg_1_4",  # 第1个手指的关节
        "rj_dg_2_1",
        "rj_dg_2_2",
        "rj_dg_2_3",
        "rj_dg_2_4",  # 第2个手指的关节
        "rj_dg_3_1",
        "rj_dg_3_2",
        "rj_dg_3_3",
        "rj_dg_3_4",  # 第3个手指的关节
        "rj_dg_4_1",
        "rj_dg_4_2",
        "rj_dg_4_3",
        "rj_dg_4_4",  # 第4个手指的关节
        "rj_dg_5_1",
        "rj_dg_5_2",
        "rj_dg_5_3",
        "rj_dg_5_4",  # 第5个手指的关节
    ]

    # 手部初始位置配置（用于初始化机器人位置）
    hand_position = [
        0.2,
        -0.1,
        0.0,
        0.1,
        0.0,  # 第1个手指的初始位置
        -0.35,
        0.3,
        0.3,
        0.3,
        0.0,  # 第2个手指的初始位置
        1.2,
        1.0,
        1.0,
        1.0,
        1.3,  # 第3个手指的初始位置
        0.1,
        1.2,
        1.2,
        1.2,
        1.57,
    ]  # 第4个手指的初始位置

    # 手部关节限制
    hand_lower_limits = [
        0,
        -24,
        -30,
        -35,
        0,
        -30,
        0,
        0,
        0,
        -15,
        20,
        25,
        25,
        20,
        0,
        0,
        68,
        68,
        68,
        89,
    ]
    hand_upper_limits = [
        30,
        35,
        30,
        24,
        60,
        0,
        70,
        30,
        30,
        30,
        90,
        90,
        90,
        90,
        90,
        20,
        70,
        70,
        70,
        90,
    ]

    # 手指 link 名称，用于过滤结构性自碰撞和惩罚跨手指干涉。
    finger_link_names = []
    for finger_index in range(1, 6):
        for joint_index in range(1, 5):
            finger_link_names.append(f"rl_dg_{finger_index}_{joint_index}")

    # 同一根手指的相邻 link 是机械结构邻接，不应该作为有意义的自碰撞接触参与求解。
    adjacent_link_collision_filter_pairs = []
    for finger_index in range(1, 6):
        for joint_index in range(1, 4):
            adjacent_link_collision_filter_pairs.append(
                (f"rl_dg_{finger_index}_{joint_index}", f"rl_dg_{finger_index}_{joint_index + 1}")
            )
        adjacent_link_collision_filter_pairs.append((f"rl_dg_{finger_index}_4", f"r{finger_index}c_sphere"))

    collision_sensor_names = finger_link_names
    collision_filter_names = {}
    for name in collision_sensor_names:
        _, _, finger_id, joint_id = name.split("_")
        joint_id = int(joint_id)
        filtered_names = []
        for other_name in finger_link_names:
            if other_name == name:
                continue

            _, _, other_finger_id, other_joint_id = other_name.split("_")
            other_joint_id = int(other_joint_id)
            if other_finger_id == finger_id and abs(other_joint_id - joint_id) <= 1:
                continue

            filtered_names.append(other_name)

        collision_filter_names[name] = filtered_names

    # 创建接触传感器配置字典
    contact_sensors = {}
    for name in collision_sensor_names:
        contact_sensors[name] = ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{name}",  # 传感器路径
            update_period=0.0,  # 更新周期
            history_length=1,  # 历史长度
            debug_vis=True,  # 是否显示调试可视化
            filter_prim_paths_expr=[
                f"/World/envs/env_.*/Robot/{contact_name}" for contact_name in collision_filter_names[name]
            ],  # 过滤的传感器路径表达式
            track_air_time=False,  # 是否跟踪空中时间
        )

    # 球体半径和质量配置
    ball_radius = 0.02  # 球体半径
    mass = 0.02  # 球体质量

    # 第一个球体配置
    ball1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball1",  # 球体路径
        spawn=sim_utils.SphereCfg(  # 球体生成配置
            radius=ball_radius,  # 球体半径
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 刚体属性
            physics_material=sim_utils.RigidBodyMaterialCfg(),  # 物理材质
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),  # 质量属性
            collision_props=sim_utils.CollisionPropertiesCfg(),  # 碰撞属性
            visual_material=sim_utils.PreviewSurfaceCfg(  # 可视化材质
                diffuse_color=(1.0, 0.0, 0.0),  # 漫反射颜色（红色）
                metallic=0.1,  # 金属度
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(  # 初始状态
            pos=(0.15691, 0.0133, 0.25702),  # 初始位置（对应tesollo 45度）
        ),
    )

    # 第二个球体配置
    ball2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball2",  # 球体路径
        spawn=sim_utils.SphereCfg(  # 球体生成配置
            radius=ball_radius,  # 球体半径
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),  # 刚体属性
            physics_material=sim_utils.RigidBodyMaterialCfg(),  # 物理材质
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),  # 质量属性
            collision_props=sim_utils.CollisionPropertiesCfg(),  # 碰撞属性
            visual_material=sim_utils.PreviewSurfaceCfg(  # 可视化材质
                diffuse_color=(1.0, 0.0, 0.0),  # 漫反射颜色（红色）
                metallic=0.1,  # 金属度
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(  # 初始状态
            pos=(0.15916, -0.0266, 0.25511),  # 初始位置（对应tesollo 45度）
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=0.75, replicate_physics=True)  # 场景配置
