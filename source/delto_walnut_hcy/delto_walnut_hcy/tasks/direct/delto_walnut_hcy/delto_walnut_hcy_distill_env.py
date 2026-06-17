# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from .delto_walnut_hcy_env import DeltoWalnutEnv
from .delto_walnut_hcy_env_cfg import DeltoWalnutEnvCfg
from .delto_walnut_hcy_collision_env import DeltoWalnutCollisionEnv
from .delto_walnut_hcy_collision_env_cfg import DeltoWalnutCollisionEnvCfg

# 版本 1：普通 curriculum env
EnvCfg = DeltoWalnutEnvCfg
Env = DeltoWalnutEnv
# 版本 2：collision env
# EnvCfg = DeltoWalnutCollisionEnvCfg
# Env = DeltoWalnutCollisionEnv

@configclass
class DeltoWalnutDistillEnvCfg(DeltoWalnutEnvCfg):
    """Use deployable observations for the actor and full observations for the critic."""

    observation_space = 53
    state_space = 79


class DeltoWalnutDistillEnv(DeltoWalnutEnv):
    """Asymmetric observation variant for learning a deployable actor.

    The actor only sees quantities that are practical to provide on the real hand:
    normalized joint positions, the two ball positions, previous action, rotation axis,
    rotation center, and ball radius. The critic keeps the original full 79-D observation,
    including simulated velocities, so PPO can still learn with richer training feedback.
    """

    cfg: DeltoWalnutDistillEnvCfg

    @property
    def num_states(self) -> int:
        """Compatibility hook for the RSL-RL wrapper's privileged-observation detection."""
        return int(self.cfg.state_space)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos[:, self.robot_joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.robot_joint_ids]

        joint_pos_norm = (
            2.0 * (joint_pos - self.hand_lower_limits) / (self.hand_upper_limits - self.hand_lower_limits + 1e-6) - 1.0
        )

        ball1_pos = self.ball1.data.root_pos_w - self.scene.env_origins
        ball2_pos = self.ball2.data.root_pos_w - self.scene.env_origins
        ball1_lin_vel = self.ball1.data.root_lin_vel_w
        ball2_lin_vel = self.ball2.data.root_lin_vel_w

        rot_axis_rep = self.rot_axis.repeat(self.scene.num_envs, 1)
        rot_center_rep = self.rot_center
        ball_radius_rep = torch.full((self.scene.num_envs, 1), self.cfg.ball_radius, device=self.device)

        actor_obs = torch.cat(
            [
                joint_pos_norm,  # N, 20
                ball1_pos,  # N, 3
                ball2_pos,  # N, 3
                self.raw_prev_actions,  # N, 20
                rot_axis_rep,  # N, 3
                rot_center_rep,  # N, 3
                ball_radius_rep,  # N, 1
            ],
            dim=-1,
        )

        critic_obs = torch.cat(
            [
                joint_pos_norm,  # N, 20
                joint_vel,  # N, 20
                ball1_pos,  # N, 3
                ball1_lin_vel,  # N, 3
                ball2_pos,  # N, 3
                ball2_lin_vel,  # N, 3
                self.raw_prev_actions,  # N, 20
                rot_axis_rep,  # N, 3
                rot_center_rep,  # N, 3
                ball_radius_rep,  # N, 1
            ],
            dim=-1,
        )

        if actor_obs.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Actor observation dim mismatch: got {actor_obs.shape[-1]}, expected {self.cfg.observation_space}."
            )
        if critic_obs.shape[-1] != self.cfg.state_space:
            raise RuntimeError(
                f"Critic observation dim mismatch: got {critic_obs.shape[-1]}, expected {self.cfg.state_space}."
            )

        return {"policy": actor_obs, "critic": critic_obs, "teacher": critic_obs}
