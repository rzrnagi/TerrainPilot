"""
Reusable telemetry logger for TerrainPilot.

Records: timestamp, base position/velocity, joint positions/velocities,
IMU (angular velocity + projected gravity), and velocity commands.
Saves to .npz. Used by open_loop_run.py and later by week4 data collection.
"""

import time
import numpy as np
from pathlib import Path


class TelemetryLogger:
    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._buf: dict[str, list] = {
            "timestamp_s":    [],
            "base_pos_w":     [],   # (3,) world XYZ
            "base_lin_vel_b": [],   # (3,) body-frame linear velocity
            "base_ang_vel_b": [],   # (3,) body-frame angular velocity (IMU)
            "projected_gravity": [], # (3,) gravity vector in body frame (IMU tilt)
            "joint_pos_rad":  [],   # (12,)
            "joint_vel_rads": [],   # (12,)
            "cmd_vel":        [],   # (3,) commanded [vx_mps, vy_mps, yaw_rps]
        }

    def record(self, robot_data, cmd_vel_tensor):
        """Append one timestep.

        Args:
            robot_data: env.unwrapped.scene["robot"].data
            cmd_vel_tensor: (1, 3) tensor of [vx, vy, yaw]
        """
        self._buf["timestamp_s"].append(time.time())
        self._buf["base_pos_w"].append(robot_data.root_pos_w[0].cpu().numpy())
        self._buf["base_lin_vel_b"].append(robot_data.root_lin_vel_b[0].cpu().numpy())
        self._buf["base_ang_vel_b"].append(robot_data.root_ang_vel_b[0].cpu().numpy())
        self._buf["projected_gravity"].append(robot_data.projected_gravity_b[0].cpu().numpy())
        self._buf["joint_pos_rad"].append(robot_data.joint_pos[0].cpu().numpy())
        self._buf["joint_vel_rads"].append(robot_data.joint_vel[0].cpu().numpy())
        self._buf["cmd_vel"].append(cmd_vel_tensor[0].cpu().numpy())

    def save(self) -> str:
        arrays = {k: np.array(v) for k, v in self._buf.items()}
        np.savez_compressed(str(self.save_path), **arrays)
        n = len(self._buf["timestamp_s"])
        print(f"[Logger] Saved {n} steps → {self.save_path}")
        return str(self.save_path)

    @staticmethod
    def load(path: str) -> dict:
        data = np.load(path)
        return {k: data[k] for k in data.files}

    @staticmethod
    def replay_stats(path: str):
        """Print a summary of a saved log for offline verification."""
        d = TelemetryLogger.load(path)
        ts = d["timestamp_s"]
        dt = np.diff(ts)
        pos = d["base_pos_w"]
        cmd = d["cmd_vel"]
        print(f"\n[Replay] {path}")
        print(f"  Steps:        {len(ts)}")
        print(f"  Duration:     {ts[-1]-ts[0]:.2f} s")
        print(f"  Step dt:      mean={dt.mean()*1000:.1f} ms  min={dt.min()*1000:.1f} ms  max={dt.max()*1000:.1f} ms")
        print(f"  Base pos:     start={pos[0]}  end={pos[-1]}")
        print(f"  Displacement: {np.linalg.norm(pos[-1]-pos[0]):.3f} m")
        print(f"  Cmd vx range: [{cmd[:,0].min():.2f}, {cmd[:,0].max():.2f}] m/s")
        print(f"  Cmd yaw range:[{cmd[:,2].min():.2f}, {cmd[:,2].max():.2f}] rad/s")
