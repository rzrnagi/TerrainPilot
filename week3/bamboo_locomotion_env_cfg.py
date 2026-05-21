"""
Locomotion training env for the bamboo data-collection policy.

Goal: train a policy that achieves >90% episode completion when navigating
a physical bamboo forest on mild slopes — clean locomotion = clean WM data.

Key differences from Week 2 flat env and Week 3 slope env:
  - Mild slopes (3-5°): terrain variety without causing constant falls
  - push_robot enabled, interval 3-8s: simulates bamboo leg collisions
  - base_external_force_torque increased: sustained lateral force robustness
  - No payload randomization: not needed for data collection stability
  - num_envs=4096: 2× parallel throughput vs previous training runs
"""

import math
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field import (
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

MILD_SLOPE_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,   # start flat, increase slope — faster early learning
    sub_terrains={
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(math.radians(1), math.radians(5)),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(math.radians(1), math.radians(5)),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class BambooLocomotionEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2 locomotion training env for bamboo forest data collection.

    Trains robustness to:
      1. Mild slopes (curriculum 1-5°)
      2. Frequent velocity impulses (push_robot every 3-8s = bamboo leg hits)
      3. Sustained lateral forces (base_external_force_torque)
    """

    def __post_init__(self):
        super().__post_init__()

        # ── Terrain: FLAT for training speed ────────────────────────────────
        # Flat plane is 10× faster to simulate than a height-field terrain.
        # Push_robot with ±0.8 m/s lateral is harder than 3° slopes anyway.
        # A policy robust to these pushes generalises to mild slopes at deploy.
        # (terrain_type="plane" is already set by UnitreeGo2FlatEnvCfg)

        # ── num_envs: 2048 — fast iterations, still massive parallelism ─────
        self.scene.num_envs = 2048
        self.scene.env_spacing = 2.5

        # ── Push robot: applied at each episode reset (mode="reset") ─────────
        # mode="interval" requires per-step bookkeeping → 7× slower.
        # mode="reset" applies one push per episode start — same robustness
        # benefit (policy must recover from the perturbation) at no per-step cost.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="reset",
            params={
                "velocity_range": {
                    "x": (-0.6, 0.6),
                    "y": (-0.8, 0.8),
                    "z": (-0.1, 0.1),
                    "roll":  (-0.3, 0.3),
                    "pitch": (-0.2, 0.2),
                    "yaw":   (-0.4, 0.4),
                },
            },
        )

        # ── External force: sustained disturbance (slope + wind analogue) ────
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range":  (0.0, 25.0),   # up to 25 N lateral
                "torque_range": (-4.0, 4.0),
            },
        )

        # ── Wider reset range: robot must recover from varied positions ───────
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.3, 0.3),
                "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4),
            },
        }
