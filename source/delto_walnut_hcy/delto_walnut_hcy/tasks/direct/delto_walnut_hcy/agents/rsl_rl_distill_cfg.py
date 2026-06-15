# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
)


@configclass
class DistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL teacher-student distillation config for the deployable actor."""

    num_steps_per_env = 16
    max_iterations = 2000
    save_interval = 200
    experiment_name = "delto_walnut"
    empirical_normalization = True
    clip_actions = 1.0

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        student_hidden_dims=[512, 512, 256, 128],
        teacher_hidden_dims=[512, 512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=5.0e-4,
        gradient_length=16,
        max_grad_norm=1.0,
    )
