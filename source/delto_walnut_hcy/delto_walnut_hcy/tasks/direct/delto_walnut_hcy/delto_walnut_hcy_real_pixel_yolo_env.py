# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab.utils import configclass

from .delto_walnut_hcy_yolo_distill_env import DeltoWalnutYoloDistillEnv, DeltoWalnutYoloDistillEnvCfg


@configclass
class DeltoWalnutRealPixelYoloDistillEnvCfg(DeltoWalnutYoloDistillEnvCfg):
    """YOLO distillation env normalized from real-camera-style ball pixels.

    ``data/example.csv`` stores ``ball_center_00..03`` as raw image pixels. This
    variant first maps simulated detections into that real pixel convention, then
    normalizes to the same ``[-1, 1]`` uv range used by ``DeltoWalnutYoloDistillEnv``.
    """

    # actor obs: joint_pos_norm(20) + normalized ball uv(4) + prev_actions(20)
    #          + rot_axis(3) + rot_center(3) + ball_radius(1)
    observation_space = 51
    state_space = 79

    # Target raw-pixel scale before normalization. This matches the replay data
    # convention used by ball_center_00..03.
    real_camera_width = 640
    real_camera_height = 480

    # Optional affine calibration from simulated camera pixels to real pixels.
    # Keep identity first; tune these if the saved debug frame and real camera
    # frame have a fixed offset or scale difference.
    real_pixel_scale: tuple[float, float] = (1.0, 1.0)
    real_pixel_offset: tuple[float, float] = (0.0, 0.0)

    debug_camera_image_path = "data/yolo_real_pixel_camera_debug.png"


class DeltoWalnutRealPixelYoloDistillEnv(DeltoWalnutYoloDistillEnv):
    """YOLO distill env with only the image-pixel normalization path changed."""

    cfg: DeltoWalnutRealPixelYoloDistillEnvCfg

    def _normalize_pixel_centers(self, centers_px: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Map source pixels to real-camera pixels, then normalize to ``[-1, 1]``.

        The parent detector calls this method before assignment. Keeping the
        method name lets the rest of ``DeltoWalnutYoloDistillEnv`` stay unchanged.
        """

        real_px = self._to_real_pixel_centers(centers_px, width, height)
        return self._normalize_real_pixel_centers(real_px)

    def _to_real_pixel_centers(self, centers_px: torch.Tensor, width: int, height: int) -> torch.Tensor:
        centers = centers_px.to(dtype=torch.float32)
        real_px = torch.empty_like(centers)

        src_w = max(float(width - 1), 1.0)
        src_h = max(float(height - 1), 1.0)
        dst_w = max(float(self.cfg.real_camera_width - 1), 1.0)
        dst_h = max(float(self.cfg.real_camera_height - 1), 1.0)

        real_px[:, 0] = centers[:, 0] * dst_w / src_w
        real_px[:, 1] = centers[:, 1] * dst_h / src_h

        scale = centers.new_tensor(self.cfg.real_pixel_scale)
        offset = centers.new_tensor(self.cfg.real_pixel_offset)
        real_px = real_px * scale + offset

        lower = centers.new_tensor((0.0, 0.0))
        upper = centers.new_tensor((float(self.cfg.real_camera_width - 1), float(self.cfg.real_camera_height - 1)))
        return torch.maximum(torch.minimum(real_px, upper), lower)

    def _normalize_real_pixel_centers(self, real_px: torch.Tensor) -> torch.Tensor:
        real_uv = torch.empty_like(real_px, dtype=torch.float32)
        real_uv[:, 0] = 2.0 * real_px[:, 0] / max(float(self.cfg.real_camera_width - 1), 1.0) - 1.0
        real_uv[:, 1] = 2.0 * real_px[:, 1] / max(float(self.cfg.real_camera_height - 1), 1.0) - 1.0
        return torch.clamp(real_uv, -1.0, 1.0)

    def _save_debug_camera_image(self, path: str | None = None):
        super()._save_debug_camera_image(path or self.cfg.debug_camera_image_path)


def register_real_pixel_yolo_task(task_id: str = "Template-Delto-Walnut-Yolo-RealPixel-Direct-v0"):
    """Register this env manually when you do not want to edit package __init__.py."""

    if task_id in gym.registry:
        return

    gym.register(
        id=task_id,
        entry_point=f"{__name__}:DeltoWalnutRealPixelYoloDistillEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}:DeltoWalnutRealPixelYoloDistillEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__package__}.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
            "rsl_rl_distill_cfg_entry_point": f"{__package__}.agents.rsl_rl_distill_cfg:DistillRunnerCfg",
        },
    )
