# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, TiledCamera, TiledCameraCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from pxr import UsdPhysics  # type: ignore

from .delto_walnut_hcy_distill_env import DeltoWalnutDistillEnv, DeltoWalnutDistillEnvCfg


@configclass
class DeltoWalnutYoloDistillEnvCfg(DeltoWalnutDistillEnvCfg):
    """Actor observes ball positions from camera detections instead of simulator positions."""

    # actor obs: joint_pos_norm(20) + ball1_uv(2) + ball2_uv(2) + prev_actions(20)
    #          + rot_axis(3) + rot_center(3) + ball_radius(1)
    observation_space = 51
    state_space = 79

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/YoloCamera",
        # Approximate the viewer direction used for videos. Tune this after checking the first camera frame.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.476, -0.006, 0.574),
            rot=(0.3827, 0.0, 0.9239, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=0.4,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        width=320,
        height=240,
    )

    # YOLO settings. Leave yolo_model_path empty to use the red-ball segmentation fallback.
    yolo_model_path = ""
    yolo_conf_threshold = 0.25
    yolo_ball_class_ids: tuple[int, ...] = ()
    yolo_device = ""

    # Fallback detector for the current simulated red balls.
    red_min_area = 12
    red_r_min = 120
    red_g_max = 120
    red_b_max = 120
    red_dominance = 1.35


class DeltoWalnutYoloDistillEnv(DeltoWalnutDistillEnv):
    """Reduced-observation environment using camera + YOLO 2D ball detections.

    The actor receives 2D normalized image positions in place of simulator ball xyz positions:
    ``[ball1_u, ball1_v, ball2_u, ball2_v]`` in ``[-1, 1]`` image coordinates.
    The critic and teacher observations keep the original full simulator state for training.
    """

    cfg: DeltoWalnutYoloDistillEnvCfg

    def __init__(self, cfg: DeltoWalnutYoloDistillEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._last_ball_uv = torch.zeros(self.num_envs, 2, 2, device=self.device)
        self._last_ball_valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._yolo_model = self._load_yolo_model()

    def _load_yolo_model(self):
        if not self.cfg.yolo_model_path:
            return None

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "DeltoWalnutYoloDistillEnv requires `ultralytics` when yolo_model_path is set. "
                "Install it or leave yolo_model_path empty to use the red-ball fallback detector."
            ) from exc

        model = YOLO(self.cfg.yolo_model_path)
        if self.cfg.yolo_device:
            model.to(self.cfg.yolo_device)
        return model

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.ball1 = RigidObject(self.cfg.ball1_cfg)
        self.ball2 = RigidObject(self.cfg.ball2_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["ball1"] = self.ball1
        self.scene.rigid_objects["ball2"] = self.ball2
        self.scene.sensors["yolo_camera"] = self._tiled_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        for name in self.cfg.ft_names:
            self.scene.sensors[name] = ContactSensor(self.cfg.contact_sensors[name])

        for env_id in range(self.num_envs):
            path_ball1 = f"/World/envs/env_{env_id}/ball1"
            path_ball2 = f"/World/envs/env_{env_id}/ball2"
            finger_ball_paths = [
                f"/World/envs/env_{env_id}/Robot/r1c_sphere",
                f"/World/envs/env_{env_id}/Robot/r2c_sphere",
                f"/World/envs/env_{env_id}/Robot/r3c_sphere",
                f"/World/envs/env_{env_id}/Robot/r4c_sphere",
                f"/World/envs/env_{env_id}/Robot/r5c_sphere",
            ]

            prim_ball1 = self.sim.stage.GetPrimAtPath(path_ball1)
            prim_ball2 = self.sim.stage.GetPrimAtPath(path_ball2)
            if prim_ball1.IsValid() and prim_ball2.IsValid():
                ball1_rel = UsdPhysics.FilteredPairsAPI.Apply(prim_ball1).CreateFilteredPairsRel()
                ball2_rel = UsdPhysics.FilteredPairsAPI.Apply(prim_ball2).CreateFilteredPairsRel()
                for finger_path in finger_ball_paths:
                    ball1_rel.AddTarget(finger_path)
                    ball2_rel.AddTarget(finger_path)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # type: ignore
        if not hasattr(self, "_last_ball_uv"):
            return
        self._last_ball_uv[env_ids] = 0.0
        self._last_ball_valid[env_ids] = False

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

        ball_uv = self._detect_ball_uv()
        ball1_uv = ball_uv[:, 0, :]
        ball2_uv = ball_uv[:, 1, :]

        rot_axis_rep = self.rot_axis.repeat(self.scene.num_envs, 1)
        rot_center_rep = self.rot_center
        ball_radius_rep = torch.full((self.scene.num_envs, 1), self.cfg.ball_radius, device=self.device)

        actor_obs = torch.cat(
            [
                joint_pos_norm,  # N, 20
                ball1_uv,  # N, 2
                ball2_uv,  # N, 2
                self.raw_prev_actions,  # N, 20
                rot_axis_rep,  # N, 3
                rot_center_rep,  # N, 3
                ball_radius_rep,  # N, 1
            ],
            dim=-1,
        )

        critic_obs = torch.cat(
            [
                joint_pos_norm,
                joint_vel,
                ball1_pos,
                ball1_lin_vel,
                ball2_pos,
                ball2_lin_vel,
                self.raw_prev_actions,
                rot_axis_rep,
                rot_center_rep,
                ball_radius_rep,
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

    def _detect_ball_uv(self) -> torch.Tensor:
        rgb = self._tiled_camera.data.output["rgb"]
        if not hasattr(self, "_last_ball_uv"):
            self._last_ball_uv = torch.zeros(self.num_envs, 2, 2, device=self.device)
            self._last_ball_valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        yolo_model = getattr(self, "_yolo_model", None)
        if yolo_model is not None:
            uv, valid = self._detect_ball_uv_with_yolo(rgb)
        else:
            uv, valid = self._detect_ball_uv_with_red_mask(rgb)

        self._last_ball_uv = torch.where(valid.unsqueeze(-1), uv, self._last_ball_uv)
        self._last_ball_valid |= valid
        return self._last_ball_uv

    def _detect_ball_uv_with_yolo(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[1], rgb.shape[2]
        images = [image for image in rgb.detach().cpu().numpy()]
        results = self._yolo_model.predict(images, verbose=False, conf=float(self.cfg.yolo_conf_threshold))

        uv = self._last_ball_uv.clone()
        valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        class_filter = set(int(class_id) for class_id in self.cfg.yolo_ball_class_ids)

        for env_id, result in enumerate(results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.detach().cpu()
            conf = boxes.conf.detach().cpu()
            cls = boxes.cls.detach().cpu().to(torch.int64)

            keep = conf >= float(self.cfg.yolo_conf_threshold)
            if class_filter:
                keep &= torch.tensor([int(class_id) in class_filter for class_id in cls.tolist()], dtype=torch.bool)
            if not torch.any(keep):
                continue

            xyxy = xyxy[keep]
            conf = conf[keep]
            order = torch.argsort(conf, descending=True)
            centers_px = 0.5 * (xyxy[order, :2] + xyxy[order, 2:])
            centers_uv = self._normalize_pixel_centers(centers_px[:2], width, height).to(self.device)
            assigned_uv, assigned_valid = self._assign_detections(env_id, centers_uv)
            uv[env_id] = assigned_uv
            valid[env_id] = assigned_valid

        return uv, valid

    def _detect_ball_uv_with_red_mask(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = rgb.shape[1], rgb.shape[2]
        uv = self._last_ball_uv.clone()
        valid = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

        red = rgb[..., 0].to(torch.float32)
        green = rgb[..., 1].to(torch.float32)
        blue = rgb[..., 2].to(torch.float32)
        mask = (
            (red >= float(self.cfg.red_r_min))
            & (green <= float(self.cfg.red_g_max))
            & (blue <= float(self.cfg.red_b_max))
            & (red >= float(self.cfg.red_dominance) * torch.maximum(green, blue))
        )

        for env_id in range(self.num_envs):
            y_idx, x_idx = torch.where(mask[env_id])
            if x_idx.numel() < int(self.cfg.red_min_area):
                continue

            x_float = x_idx.to(torch.float32)
            split_x = torch.median(x_float)
            centers = []
            for side_mask in (x_float <= split_x, x_float > split_x):
                if torch.count_nonzero(side_mask) < int(self.cfg.red_min_area):
                    continue
                centers.append(
                    torch.stack(
                        [
                            x_float[side_mask].mean(),
                            y_idx[side_mask].to(torch.float32).mean(),
                        ]
                    )
                )

            if len(centers) == 1:
                centers_px = torch.stack(centers, dim=0)
            elif len(centers) >= 2:
                centers_px = torch.stack(centers[:2], dim=0)
            else:
                continue

            centers_uv = self._normalize_pixel_centers(centers_px, width, height).to(self.device)
            assigned_uv, assigned_valid = self._assign_detections(env_id, centers_uv)
            uv[env_id] = assigned_uv
            valid[env_id] = assigned_valid

        return uv, valid

    def _normalize_pixel_centers(self, centers_px: torch.Tensor, width: int, height: int) -> torch.Tensor:
        centers_uv = torch.empty_like(centers_px, dtype=torch.float32)
        centers_uv[:, 0] = 2.0 * centers_px[:, 0] / max(width - 1, 1) - 1.0
        centers_uv[:, 1] = 2.0 * centers_px[:, 1] / max(height - 1, 1) - 1.0
        return torch.clamp(centers_uv, -1.0, 1.0)

    def _assign_detections(self, env_id: int, detections_uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assigned = self._last_ball_uv[env_id].clone()
        valid = torch.zeros(2, dtype=torch.bool, device=self.device)
        count = detections_uv.shape[0]

        if count == 0:
            return assigned, valid

        if count == 1:
            target_id = 0
            if torch.any(self._last_ball_valid[env_id]):
                distances = torch.linalg.norm(self._last_ball_uv[env_id] - detections_uv[0], dim=-1)
                target_id = int(torch.argmin(distances).item())
            assigned[target_id] = detections_uv[0]
            valid[target_id] = True
            return assigned, valid

        det = detections_uv[:2]
        if torch.all(self._last_ball_valid[env_id]):
            direct_cost = torch.linalg.norm(det[0] - self._last_ball_uv[env_id, 0]) + torch.linalg.norm(
                det[1] - self._last_ball_uv[env_id, 1]
            )
            swap_cost = torch.linalg.norm(det[1] - self._last_ball_uv[env_id, 0]) + torch.linalg.norm(
                det[0] - self._last_ball_uv[env_id, 1]
            )
            if swap_cost < direct_cost:
                det = det[[1, 0]]
        else:
            det = det[torch.argsort(det[:, 0])]

        assigned[:] = det
        valid[:] = True
        return assigned, valid


def register_yolo_distill_task(task_id: str = "Template-Delto-Walnut-Yolo-Direct-v0") -> None:
    """Register the YOLO-camera distillation environment when this module is imported manually."""
    if task_id in gym.registry:
        return

    module_name = __name__
    package_name = module_name.rsplit(".", 1)[0]
    gym.register(
        id=task_id,
        entry_point=f"{module_name}:DeltoWalnutYoloDistillEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{module_name}:DeltoWalnutYoloDistillEnvCfg",
            "rsl_rl_cfg_entry_point": f"{package_name}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
            "rsl_rl_distill_cfg_entry_point": f"{package_name}.agents.rsl_rl_distill_cfg:DistillRunnerCfg",
        },
    )


__all__ = ["DeltoWalnutYoloDistillEnv", "DeltoWalnutYoloDistillEnvCfg", "register_yolo_distill_task"]
