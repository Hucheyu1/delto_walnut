from __future__ import annotations

import os
import pickle
from typing import Any


class DataRecorder:
    """Episode-level pickle recorder for Isaac Lab tests.

    Save one .pkl file whenever the environment resets. The saved data is a dict
    whose values are lists. Each list element corresponds to one recorded frame.
    """

    def __init__(
        self,
        save_dir: str = "/home/amlrobotics/hcy_ws/delto_walnut_hcy/data",
        prefix: str = "data_0615",
        enable: bool = False,
    ):
        self.save_dir = save_dir
        self.prefix = prefix
        self.enable = enable
        self.reset_count = 0

        os.makedirs(self.save_dir, exist_ok=True)
        self.data = self._new_buffer()

    def _new_buffer(self) -> dict[str, list[Any]]:
        return {
            "frame": [],
            "actions": [],
            "target_pos": [],
            "obs_joint_pos": [],
            "ball1_pos": [],
            "ball2_pos": [],
        }

    @staticmethod
    def _to_list(value: Any) -> Any:
        """Convert torch / numpy data to pickle-friendly Python lists."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    def record(
        self,
        *,
        frame: Any = None,
        actions: Any = None,
        target_pos: Any = None,
        obs_joint_pos: Any = None,
        ball1_pos: Any = None,
        ball2_pos: Any = None,
    ) -> None:
        """Record one simulation/control frame."""
        if not self.enable:
            return

        self.data["frame"].append(self._to_list(frame))
        self.data["actions"].append(self._to_list(actions))
        self.data["target_pos"].append(self._to_list(target_pos))
        self.data["obs_joint_pos"].append(self._to_list(obs_joint_pos))
        self.data["ball1_pos"].append(self._to_list(ball1_pos))
        self.data["ball2_pos"].append(self._to_list(ball2_pos))

    def save_and_reset(self) -> str | None:
        """Save current buffer to disk and clear memory.

        Returns:
            Saved path if a file was written. Otherwise None.
        """
        if not self.enable:
            return None

        # Avoid saving empty files during the first reset.
        if len(self.data["frame"]) == 0:
            return None

        save_path = os.path.join(self.save_dir, f"{self.prefix}_{self.reset_count}.pkl")
        with open(save_path, "wb") as file:
            pickle.dump(self.data, file)

        print(f"[DataRecorder] saved: {save_path}")

        self.reset_count += 1
        self.data = self._new_buffer()
        return save_path

    def save_final(self) -> str | None:
        """Call this before program exit to avoid losing the final unfinished episode."""
        return self.save_and_reset()
