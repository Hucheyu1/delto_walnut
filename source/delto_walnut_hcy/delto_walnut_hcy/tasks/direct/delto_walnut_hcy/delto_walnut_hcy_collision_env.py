# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from pxr import UsdPhysics  # type: ignore

from .delto_walnut_hcy_collision_env_cfg import DeltoWalnutCollisionEnvCfg

np.set_printoptions(precision=3, suppress=True)  # print禁用科学计数法，保留3位小数


class DeltoWalnutCollisionEnv(DirectRLEnv):
    """
    DeltoWalnutEnv 是一个强化学习环境，用于控制机械手抓取并旋转两个坚果球。

    该环境实现了基于物理的仿真，通过控制机械手关节来实现对两个球体的精确控制。
    环境包含奖励函数，用于评估机械手是否成功将两个球体保持在特定的旋转约束下。

    Attributes:
        cfg: 环境配置对象，包含机器人、球体等配置参数
    """

    cfg: DeltoWalnutCollisionEnvCfg

    def __init__(self, cfg: DeltoWalnutCollisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.global_env_step = 0
        N = self.scene.num_envs

        # 获取关节引用
        self.robot_joint_ids, _ = self.robot.find_joints(self.cfg.hand_joint_names)

        # 动作/关节限位
        self.action_scale = cfg.action_scale
        self.hand_lower_limits = torch.tensor(cfg.hand_lower_limits, device=self.device) * np.pi / 180
        self.hand_upper_limits = torch.tensor(cfg.hand_upper_limits, device=self.device) * np.pi / 180

        # 初始位姿
        self.hand_start_position = torch.tensor(cfg.hand_position, device=self.device)

        # 球相对于手的位姿
        self.ball1_to_robot_pos = torch.tensor([[0.0452, -0.0133, 0.1767]], device=self.device)
        self.ball1_to_robot_quat = torch.tensor(
            [[-4.0384e-08, -9.2388e-01, -1.6727e-08, -3.8268e-01]], device=self.device
        )
        self.ball2_to_robot_pos = torch.tensor([[0.0454, 0.0266, 0.1796]], device=self.device)
        self.ball2_to_robot_quat = torch.tensor(
            [[-4.0384e-08, -9.2388e-01, -1.6727e-08, -3.8268e-01]], device=self.device
        )

        # 创建动作缓存：当前动作、上一步动作（用于计算平滑度）、目标位置（用于累加）
        self.raw_actions = torch.zeros(N, cfg.action_space, device=self.device)
        self.raw_prev_actions = torch.zeros_like(self.raw_actions)
        self.target_pos = self.hand_start_position.clone().unsqueeze(0).repeat(N, 1)

        # 提取两个核桃在世界坐标系下的默认初始位置
        self.ball1_default_pos = self.ball1.data.default_root_state[:, 0:3]
        self.ball2_default_pos = self.ball2.data.default_root_state[:, 0:3]
        self.rot_center = (self.ball1_default_pos + self.ball2_default_pos) / 2  # (N,3)

        # 定义理想的旋转轴：一个斜向下 45 度的向量向量 [-0.707, 0, -0.707]
        self.rot_axis = torch.tensor([-0.707, 0, -0.707], device=self.device).unsqueeze(0)
        # self.rot_axis = torch.tensor([-math.cos(math.pi/3), 0, -math.sin(math.pi/6)], device=self.device).unsqueeze(0)

        # 日志统计
        self.extras["log"] = {
            "r_rot": torch.zeros(N, device=self.device),
            "r_sym": torch.zeros(N, device=self.device),
            "r_tan": torch.zeros(N, device=self.device),
            "r_axis": torch.zeros(N, device=self.device),
            "r_dropped": torch.zeros(N, device=self.device),
            "r_smooth": torch.zeros(N, device=self.device),
            "r_omega": torch.zeros(N, device=self.device),
            "r_torque": torch.zeros(N, device=self.device),
            "r_collision": torch.zeros(N, device=self.device),
            "total_reward": torch.zeros(N, device=self.device),
        }

        print(f"max_episode_length: {self.max_episode_length}")
        self.pt_ct = 0

        # self.data = dict()
        # self.data['frame'] = list()
        # self.data['actions'] = list()
        # self.data['target_pos'] = list()
        # self.data['obs_joint_pos'] = list()
        self.reset_count = 0

    def _setup_scene(self):
        # 实例化机器人和两个球体
        self.robot = Articulation(self.cfg.robot_cfg)
        self.ball1 = RigidObject(self.cfg.ball1_cfg)
        self.ball2 = RigidObject(self.cfg.ball2_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # 依照 num_envs 克隆并行环境
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # 将实体注册到场景的字典中以便后续管理
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball1"] = self.ball1
        self.scene.rigid_objects["ball2"] = self.ball2
        # 添加光照和接触传感器（用于检测挤压碰撞力）
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # add contact sensors
        for name in self.cfg.collision_sensor_names:
            self.scene.sensors[name] = ContactSensor(self.cfg.contact_sensors[name])
        # -------- 底层物理碰撞过滤 --------
        for env_id in range(self.num_envs):
            path_ball1 = f"/World/envs/env_{env_id}/ball1"
            path_ball2 = f"/World/envs/env_{env_id}/ball2"
            # 获取指尖 (r1c_sphere 到 r5c_sphere) 的 USD 路径
            finger_ball_path = list()
            finger_ball_path.append(f"/World/envs/env_{env_id}/Robot/r1c_sphere")
            finger_ball_path.append(f"/World/envs/env_{env_id}/Robot/r2c_sphere")
            finger_ball_path.append(f"/World/envs/env_{env_id}/Robot/r3c_sphere")
            finger_ball_path.append(f"/World/envs/env_{env_id}/Robot/r4c_sphere")
            finger_ball_path.append(f"/World/envs/env_{env_id}/Robot/r5c_sphere")
            prim_ball1 = self.sim.stage.GetPrimAtPath(path_ball1)
            prim_ball2 = self.sim.stage.GetPrimAtPath(path_ball2)
            # print(f"prim_ball1: {prim_ball1}")
            # print(f"prim_ball2: {prim_ball2}")
            # 使用 USD 原生 API (UsdPhysics.FilteredPairsAPI)
            # 强制让球 1 和球 2 忽略与指尖（r*c_sphere）的物理碰撞
            if prim_ball1.IsValid():
                ball1_api = UsdPhysics.FilteredPairsAPI.Apply(prim_ball1)
                ball2_api = UsdPhysics.FilteredPairsAPI.Apply(prim_ball2)
                ball1_rel = ball1_api.CreateFilteredPairsRel()
                ball2_rel = ball2_api.CreateFilteredPairsRel()
                for finger_path in finger_ball_path:
                    ball1_rel.AddTarget(finger_path)
                    ball2_rel.AddTarget(finger_path)

            # 同一手指相邻 link 的碰撞属于结构邻接干涉，直接在 PhysX 层过滤。
            for link_a, link_b in self.cfg.adjacent_link_collision_filter_pairs:
                path_a = f"/World/envs/env_{env_id}/Robot/{link_a}"
                path_b = f"/World/envs/env_{env_id}/Robot/{link_b}"
                prim_a = self.sim.stage.GetPrimAtPath(path_a)
                prim_b = self.sim.stage.GetPrimAtPath(path_b)
                if prim_a.IsValid() and prim_b.IsValid():
                    rel_a = UsdPhysics.FilteredPairsAPI.Apply(prim_a).CreateFilteredPairsRel()
                    rel_b = UsdPhysics.FilteredPairsAPI.Apply(prim_b).CreateFilteredPairsRel()
                    rel_a.AddTarget(path_b)
                    rel_b.AddTarget(path_a)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.global_env_step += 1

        # self.data['actions'].append(actions.tolist())
        # 先保存旧动作
        self.raw_prev_actions.copy_(self.raw_actions)
        # 再更新当前动作
        self.raw_actions = torch.clamp(actions, -1.0, 1.0)
        # 【重要】增量式控制 (Delta-position Control)
        # 目标位置 = 当前目标位置 + 缩放系数 * 网络输出动作
        self.target_pos = self.target_pos + self.action_scale * self.raw_actions
        self.target_pos = torch.clamp(self.target_pos, self.hand_lower_limits, self.hand_upper_limits)

    def _apply_action(self) -> None:
        # 将最终算出的目标位置下发给机器人的 PD 控制器
        self.robot.set_joint_position_target(self.target_pos, joint_ids=self.robot_joint_ids)
        # self.data['frame'].append(self.episode_length_buf.tolist())
        # self.data['target_pos'].append(self.target_pos.tolist())

    def _get_observations(self) -> dict:
        # ---------------- robot 获取关节真实物理位置和速度----------------
        joint_pos = self.robot.data.joint_pos[:, self.robot_joint_ids]  # (N, dof)
        joint_vel = self.robot.data.joint_vel[:, self.robot_joint_ids]  # (N, dof)

        # 将关节位置从物理弧度值归一化到 [-1, 1] 的区间，有利于神经网络加速收敛
        joint_pos_norm = (
            2.0 * (joint_pos - self.hand_lower_limits) / (self.hand_upper_limits - self.hand_lower_limits + 1e-6) - 1.0
        )

        # ---------------- object 变成每个并行环境自己的局部坐标，避免不同 env 的世界坐标偏移污染观测。----------------
        ball1_pos = self.ball1.data.root_pos_w - self.scene.env_origins
        ball2_pos = self.ball2.data.root_pos_w - self.scene.env_origins
        ball1_lin_vel = self.ball1.data.root_lin_vel_w
        ball2_lin_vel = self.ball2.data.root_lin_vel_w

        # # 沿着 batch 维度复制不变量（旋转轴、中心点、半径）
        rot_axis_rep = self.rot_axis.repeat(self.scene.num_envs, 1)  # [num_envs, 3]
        rot_center_rep = self.rot_center  # [num_envs, 3]
        ball_radius_rep = torch.tensor(self.cfg.ball_radius, device=self.device).repeat(
            self.scene.num_envs, 1
        )  # [num_envs, 1]

        obs = torch.cat(
            [
                joint_pos_norm,  # N, 20  关节位置归一化值
                joint_vel,  # N, 20  关节速度
                ball1_pos,  # N, 3   球1位置坐标
                ball1_lin_vel,  # N, 3   球1线速度
                ball2_pos,  # N, 3   球2位置坐标
                ball2_lin_vel,  # N, 3   球2线速度
                self.raw_prev_actions,  # N, 20  前一时刻的动作
                rot_axis_rep,  # N, 3   旋转轴表示
                rot_center_rep,  # N, 3   旋转中心表示
                ball_radius_rep,  # N, 1   球半径表示
            ],
            dim=-1,
        )
        # self.data['obs_joint_pos'].append(joint_pos.tolist())
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # self.pt_ct += 1
        obj_ball1_z = self.ball1.data.root_pos_w[:, 2]
        obj_ball2_z = self.ball2.data.root_pos_w[:, 2]
        terminated = (obj_ball1_z < self.cfg.drop_height_threshold) | (obj_ball2_z < self.cfg.drop_height_threshold)
        truncated = self.episode_length_buf >= (self.max_episode_length - 1)
        # if self.pt_ct == 10:
        #     print("run time:",self.episode_length_buf[0], self.max_episode_length)
        #     self.pt_ct = 0
        return terminated, truncated

    # 在仿真环境每次重置时，将当前环境状态保存到文件中，并重置各种状态变量、关节位置、机器人和球体的位姿等
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # with open('/home/amlrobotics/hcy_ws/record_data/data_0528_' + str(self.reset_count) + '.pkl', 'wb') as file:
        #     pickle.dump(self.data, file)
        #     self.reset_count += 1
        #     for key in self.data:
        #         self.data[key].clear()

        self.reset_count += 1
        # for key in self.data:
        #     self.data[key].clear()
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)

        # 重置日志
        for k in self.extras["log"]:
            self.extras["log"][k][env_ids] = 0.0

        # 重置 reward 相关
        self.target_pos[env_ids] = self.hand_start_position.clone()
        self.raw_actions[env_ids] = 0.0
        self.raw_prev_actions[env_ids] = 0.0

        # 重置关节
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -------- 手和球的 pose 重置（都先在 env 局部坐标系中处理） --------
        # 这些 default_root_state 是在 env 局部坐标系下的
        robot_root_state = self.robot.data.default_root_state[env_ids].clone()  # (N, 13)
        ball1_root_state = self.ball1.data.default_root_state[env_ids].clone()  # (N, 13)
        ball2_root_state = self.ball2.data.default_root_state[env_ids].clone()  # (N, 13)

        # -------- 统一加 env_origins，写回模拟 --------
        env_origins = self.scene.env_origins[env_ids]  # (N, 3)

        robot_root_state[:, :3] += env_origins
        ball1_root_state[:, :3] += env_origins
        ball2_root_state[:, :3] += env_origins

        self.robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root_state[:, 7:], env_ids)
        self.ball1.write_root_state_to_sim(ball1_root_state, env_ids)
        self.ball2.write_root_state_to_sim(ball2_root_state, env_ids)

    def _smooth_ramp(self, x: float, start: float, end: float) -> float:
        """Smoothly increase from 0 to 1 between start and end."""
        if x <= start:
            return 0.0
        if x >= end:
            return 1.0

        p = (x - start) / (end - start)
        return p * p * (3.0 - 2.0 * p)

    def _get_curriculum_weights(self) -> dict[str, float]:
        """Two-stage curriculum.

        Stage 1:
            保持双球几何结构 + 转起来 + 不掉落。

        Stage 2:
            在已经能完成主任务的基础上，逐渐加入动作平滑、力矩、碰撞惩罚。
        """

        if not getattr(self.cfg, "enable_curriculum", True):
            return {
                "rot": 10e-3,
                "sym": 0.5 * 10e-3,
                "tan": 1.0 * 10e-5,
                "dropped": 1.0,
                "omega": 2.0 * 10e-4,
                "axis": 0.5 * 10e-2,
                "smooth": 0.5 * 10e-6,
                "torque": 1.0 * 10e-4,
                "collision": 5.0e-4,
            }

        # RSL-RL 中一次 policy iteration 大约对应 num_steps_per_env 个 env step
        steps_per_iter = getattr(self.cfg, "curriculum_steps_per_iter", 16)
        cur_iter = self.global_env_step / max(1, steps_per_iter)

        quality_start_iter = getattr(self.cfg, "quality_start_iter", 500)
        quality_end_iter = getattr(self.cfg, "quality_end_iter", 1000)

        # 主任务权重：核奖励版本下不要再用 sym=100
        w_rot = 10e-3
        w_sym = 0.5 * 10e-3
        w_tan = 1.0 * 10e-5
        w_dropped = 1.0
        w_omega = 2.0 * 10e-4
        w_axis = 0.5 * 10e-2

        # 动作质量课程：从 quality_start_iter 到 quality_end_iter 平滑打开
        q = self._smooth_ramp(cur_iter, quality_start_iter, quality_end_iter)

        # 前期保留一点动作平滑，避免完全乱抖；后期逐渐增强
        w_smooth = 0.5 * 10e-6
        w_torque = 1.0 * 10e-4

        # 手指自碰撞从一开始就轻微惩罚，后期增强，避免先学会“互相顶住”再难以改掉。
        w_collision = (0.25 + 0.75 * q) * 5.0e-4

        return {
            "rot": w_rot,
            "sym": w_sym,
            "tan": w_tan,
            "dropped": w_dropped,
            "omega": w_omega,
            "axis": w_axis,
            "smooth": w_smooth,
            "torque": w_torque,
            "collision": w_collision,
        }

    def _get_rewards(self) -> torch.Tensor:

        cur_w = self._get_curriculum_weights()
        # -------- 基础变量 --------
        ball1_pos = self.ball1.data.root_pos_w - self.scene.env_origins
        ball2_pos = self.ball2.data.root_pos_w - self.scene.env_origins
        ball1_lin_vel = self.ball1.data.root_lin_vel_w
        ball2_lin_vel = self.ball2.data.root_lin_vel_w

        radius = self.cfg.ball_radius
        center = self.rot_center
        axis = self.rot_axis  # (1, 3)

        # -------- 1. 半径约束 --------
        vec1 = ball1_pos - center
        vec2 = ball2_pos - center
        dist1 = vec1.norm(dim=-1)
        dist2 = vec2.norm(dim=-1)
        rew_radius = -((dist1 - radius) ** 2 + (dist2 - radius) ** 2)
        r_rot = cur_w["rot"] * rew_radius

        # -------- 2. 中心对称约束 --------
        # symmetry_err = (ball1_pos + ball2_pos - 2 * center).norm(dim=-1)
        # rew_symmetry = -(symmetry_err ** 2)
        # r_sym = self.cfg.w_symmetry * rew_symmetry
        p1 = ball1_pos - center
        p2 = ball2_pos - center
        # 轴向 / 垂直分解
        p1_axis = (p1 * axis).sum(dim=-1, keepdim=True)
        p2_axis = (p2 * axis).sum(dim=-1, keepdim=True)
        p1_perp = p1 - p1_axis * axis
        p2_perp = p2 - p2_axis * axis
        # 2.1. 中心对称（原有）
        sym_center_err = (ball1_pos + ball2_pos - 2 * center).norm(dim=-1)
        rew_sym_center = -(sym_center_err**2)
        # 2.2. 轴向位置一致
        sym_axis_err = (p1_axis - p2_axis).squeeze(-1)
        rew_sym_axis = -(sym_axis_err**2)
        # 2.3. 垂直镜像对称
        sym_perp_err = (p1_perp + p2_perp).norm(dim=-1)
        rew_sym_perp = -(sym_perp_err**2)
        rew_symmetry = rew_sym_center + 10 * rew_sym_perp
        r_sym = cur_w["sym"] * rew_symmetry

        # -------- 3. 切向约束（速度与径向垂直）--------
        # 使用点乘判断是否垂直
        dot1 = (ball1_lin_vel * vec1).sum(dim=-1)
        dot2 = (ball2_lin_vel * vec2).sum(dim=-1)
        norm_vel1 = ball1_lin_vel.norm(dim=-1)
        norm_vel2 = ball2_lin_vel.norm(dim=-1)
        norm_vec1 = vec1.norm(dim=-1)
        norm_vec2 = vec2.norm(dim=-1)

        tangential_err1 = torch.abs(dot1) / (norm_vel1 * norm_vec1 + 1e-6)
        tangential_err2 = torch.abs(dot2) / (norm_vel2 * norm_vec2 + 1e-6)
        rew_tangential = -(tangential_err1**2 + tangential_err2**2)
        r_tan = cur_w["tan"] * rew_tangential

        # -------- 4. 绕轴方向约束 --------
        desired_v1 = torch.cross(axis, vec1, dim=-1)
        desired_v2 = torch.cross(axis, vec2, dim=-1)

        desired_v1 = desired_v1 / (desired_v1.norm(dim=-1, keepdim=True) + 1e-6)
        desired_v2 = desired_v2 / (desired_v2.norm(dim=-1, keepdim=True) + 1e-6)

        v1_norm = ball1_lin_vel / (norm_vel1.unsqueeze(-1) + 1e-6)
        v2_norm = ball2_lin_vel / (norm_vel2.unsqueeze(-1) + 1e-6)

        # 速度太小时不奖励“方向正确”，避免低频策略停在几乎不转的局部最优。
        speed_gate1 = 1.0 - torch.exp(-((norm_vel1 / self.cfg.speed_kernel_sigma) ** 2))
        speed_gate2 = 1.0 - torch.exp(-((norm_vel2 / self.cfg.speed_kernel_sigma) ** 2))
        rew_axis = speed_gate1 * (v1_norm * desired_v1).sum(dim=-1) + speed_gate2 * (
            v2_norm * desired_v2
        ).sum(dim=-1)
        r_axis = cur_w["axis"] * rew_axis

        # ---------------- 6. 掉落惩罚 ----------------
        obj_ball1_z = self.ball1.data.root_pos_w[:, 2]
        obj_ball2_z = self.ball2.data.root_pos_w[:, 2]
        terminated = (obj_ball1_z < self.cfg.drop_height_threshold + 0.005) | (
            obj_ball2_z < self.cfg.drop_height_threshold + 0.005
        )
        dropped_now = terminated & (self.episode_length_buf > 0)  # 刚掉落的那一步”给惩罚（避免连续给很多步）
        r_dropped = -dropped_now.float()
        r_dropped = cur_w["dropped"] * r_dropped

        # ---------------- 7. 关节平滑惩罚（动作变化） ----------------
        action_diff = self.raw_actions - self.raw_prev_actions
        r_smooth = -torch.sum(action_diff**2, dim=-1)
        r_smooth = cur_w["smooth"] * r_smooth

        # ---------------- 8. 速度奖励 ----------------
        r1 = vec1  # ball1_pos - center
        r2 = vec2  # ball2_pos - center
        # 去掉 r 在 axis 方向上的分量 -> 投影到垂直于 axis 的平面
        r1_parallel = (r1 * axis).sum(dim=-1, keepdim=True) * axis  # (N,3)
        r2_parallel = (r2 * axis).sum(dim=-1, keepdim=True) * axis
        r1_perp = r1 - r1_parallel
        r2_perp = r2 - r2_parallel
        r1_perp_norm_sq = (r1_perp * r1_perp).sum(dim=-1)  # |r_perp|^2
        r2_perp_norm_sq = (r2_perp * r2_perp).sum(dim=-1)
        # axis · (r × v) ，为绕该轴的转动“力矩”类似量
        # 注意 cross 的顺序：r × v
        cross1 = torch.cross(r1, ball1_lin_vel, dim=-1)
        cross2 = torch.cross(r2, ball2_lin_vel, dim=-1)
        axis_dot_cross1 = (axis * cross1).sum(dim=-1)
        axis_dot_cross2 = (axis * cross2).sum(dim=-1)
        # 角速度（正负表示绕轴方向）
        omega1 = axis_dot_cross1 / (r1_perp_norm_sq + 1e-6)
        omega2 = axis_dot_cross2 / (r2_perp_norm_sq + 1e-6)

        rew_omega = torch.relu(omega1) + torch.relu(omega2)
        r_omega = cur_w["omega"] * rew_omega

        # ---------------- 9. 关节力矩惩罚（基础款） ----------------
        tau = self.robot.root_physx_view.get_link_incoming_joint_force()  # [N_env, N_link, 6]
        f = tau[..., :3]  # force
        m = tau[..., 3:]  # torque
        f_norm = torch.linalg.norm(f, dim=-1)  # [N_env, N_link]
        m_norm = torch.linalg.norm(m, dim=-1)  # [N_env, N_link]
        # print("norms", f_norm.sum(dim=-1), m_norm.sum(dim=-1))
        # 超参数：按任务调
        wf = 1e-2  # force 权重
        wm = 1e-1  # torque 权重
        penalty_per_link = wf * f_norm + wm * m_norm  # [N_env, N_link]
        penalty = penalty_per_link.sum(dim=-1)  # [N_env]
        r_torque = -cur_w["torque"] * penalty

        # ---------------- 10. 指间碰撞惩罚 ----------------
        collision_penalty = torch.zeros(self.num_envs, device=self.device)
        for name in self.cfg.collision_sensor_names:
            force_data = self.scene.sensors[name].data.net_forces_w  # [num_envs, 1, 3]
            force_norm = torch.linalg.norm(force_data, dim=-1)  # [num_envs, 1]
            collision_penalty += force_norm.squeeze(-1)  # [num_envs]
            # print(f"{name}: {force_norm}")
            # print(f"{name}: {force_norm.shape}")
        collision_penalty = 0.5 * collision_penalty  # 接触通常会被两个 link 的传感器各统计一次
        # print(f"{name}: {collision_penalty.shape}")
        r_collision = -cur_w["collision"] * collision_penalty

        # reward = r_sym  * 100 + r_dropped + r_omega + r_axis
        reward = r_rot + r_sym * 100 + r_tan + r_dropped + r_axis + r_smooth + r_torque + r_omega + r_collision
        # r_rot 接近 0  → 球基本保持在目标半径附近
        # r_sym 接近 0  → 双球几何结构基本保持住
        # r_tan 接近 0  → 速度方向基本没有明显径向乱飞
        self.extras["log"]["r_rot"] = r_rot
        self.extras["log"]["r_sym"] = r_sym
        self.extras["log"]["r_tan"] = r_tan
        self.extras["log"]["r_axis"] = r_axis
        self.extras["log"]["r_dropped"] = r_dropped
        self.extras["log"]["r_smooth"] = r_smooth
        self.extras["log"]["r_omega"] = r_omega
        self.extras["log"]["r_torque"] = r_torque
        self.extras["log"]["r_collision"] = r_collision
        self.extras["log"]["total_reward"] = reward

        return reward
