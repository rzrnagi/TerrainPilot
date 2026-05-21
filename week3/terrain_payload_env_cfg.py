"""
Week 3 environment config: sloped terrain + payload randomization.

Extends the Go2 rough env with:
  - Slope-only terrain (5-15°) replacing the default mixed rough terrain
  - Payload mass randomization: 0–8 kg added to base
  - COM offset randomization: ±10 cm in XY, ±2 cm in Z
  - num_envs=2048 for training (reduce to 256 for eval)
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field import (
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
)
# Inherit from FLAT env so obs dim (48) matches the Week 2 checkpoint.
# The flat env has no height scanner — we add terrain but keep the same obs space.
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg


# Terrain with only slopes in 5–15° range.
# Alternating pyramid (slope up) and inverted pyramid (slope down) gives the
# robot experience with both ascending and descending faces.
SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(math.radians(5), math.radians(15)),
            platform_width=1.5,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(math.radians(5), math.radians(15)),
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)


@configclass
class UnitreeGo2SlopePayloadEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2 on sloped terrain with payload randomization.

    Inherits from the FLAT env (48-dim obs, no height scanner) so the
    Week 2 checkpoint can be loaded for fine-tuning without obs mismatch.
    We add sloped terrain and payload mass/COM randomization on top.
    """

    def __post_init__(self):
        super().__post_init__()

        # --- Terrain: swap plane for slope-only generator ---
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = SLOPE_TERRAIN_CFG

        # --- Payload: 0–8 kg added mass on base ---
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (0.0, 8.0),
                "operation": "add",
            },
        )

        # --- COM offset: ±10 cm XY, ±2 cm Z ---
        self.events.base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (-0.02, 0.02)},
            },
        )

        # --- Training scale ---
        self.scene.num_envs = 2048
        self.scene.env_spacing = 3.0

        # Increase orientation penalty to handle slopes
        self.rewards.flat_orientation_l2.weight = -3.0


@configclass
class UnitreeGo2SlopePayloadEnvCfg_EVAL(UnitreeGo2SlopePayloadEnvCfg):
    """Smaller version for evaluation comparisons."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.scene.env_spacing = 3.0
        self.scene.terrain.terrain_generator.num_rows = 4
        self.scene.terrain.terrain_generator.num_cols = 4
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
