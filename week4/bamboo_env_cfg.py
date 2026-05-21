"""
Bamboo-forest scene config for LeWM data collection (v3 — final).

Design rationale:
  - MILD slopes (3-5°): terrain variety for realistic visual dynamics,
    mild enough that the flat-trained policy survives random resets reliably
  - PHYSICAL bamboo: robot must navigate around stems — LeWM learns real
    obstacle avoidance visual dynamics
  - base_contact termination DISABLED: bamboo stem contact would fire it;
    fall detection handled by height threshold in collection script
  - 5 forest layouts via different RNG seeds for generalization
  - Stem positions exported so collection script can compute avoidance
    analytically without a depth sensor
"""

import math
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.height_field import (
    HfPyramidSlopedTerrainCfg,
    HfInvertedPyramidSlopedTerrainCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

# ── Bamboo forest parameters ──────────────────────────────────────────────────
STEM_RADIUS  = 0.06   # m
STEM_HEIGHT  = 3.5    # m
STEM_COLOR   = (0.35, 0.55, 0.15)
N_STEMS      = 20
GRID_EXTENT  = 10.0   # arena half-width (stems placed in ±GRID_EXTENT)
CLEAR_RADIUS = 2.0    # no stems within this radius of arena origin
LAYOUT_SEEDS = [0, 42, 137, 256, 999]

# Mild slope terrain: 3–5° — real terrain variation, stable for flat policy
MILD_SLOPE_TERRAIN = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=6,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(math.radians(3), math.radians(5)),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(math.radians(3), math.radians(5)),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
# ─────────────────────────────────────────────────────────────────────────────


def generate_stem_positions(seed: int) -> np.ndarray:
    """Return (N_STEMS, 2) array of stem (x, y) positions."""
    rng = np.random.default_rng(seed)
    positions = []
    while len(positions) < N_STEMS:
        x = rng.uniform(-GRID_EXTENT, GRID_EXTENT)
        y = rng.uniform(-GRID_EXTENT, GRID_EXTENT)
        if math.hypot(x, y) >= CLEAR_RADIUS:
            positions.append([float(x), float(y)])
    return np.array(positions)


ALL_LAYOUTS = {seed: generate_stem_positions(seed) for seed in LAYOUT_SEEDS}


def build_bamboo_scene_cfg(scene_cfg, seed: int = 0) -> None:
    """Attach camera + physical bamboo stems to an InteractiveSceneCfg."""
    positions = generate_stem_positions(seed)

    scene_cfg.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,   # 10 Hz
        height=240, width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    for i, (x, y) in enumerate(positions):
        setattr(
            scene_cfg,
            f"bamboo_{i:02d}",
            AssetBaseCfg(
                prim_path=f"/World/bamboo/stem_{i:02d}",
                spawn=sim_utils.CylinderCfg(
                    radius=STEM_RADIUS,
                    height=STEM_HEIGHT,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=STEM_COLOR, metallic=0.0
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(x, y, STEM_HEIGHT / 2.0)
                ),
            ),
        )


@configclass
class UnitreeGo2BambooEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2 in a physical bamboo forest on mild slope terrain.

    Mild slopes (3-5°) provide terrain variety for realistic visual dynamics
    while keeping the policy stable from random reset positions.
    Physical bamboo forces the robot to navigate around stems.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 60.0

        # Override flat plane with mild slope terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = MILD_SLOPE_TERRAIN
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.max_init_terrain_level = None

        build_bamboo_scene_cfg(self.scene, seed=LAYOUT_SEEDS[0])

        self.events.reset_base.params = {
            "pose_range": {
                "x": (-4.0, 4.0), "y": (-4.0, 4.0), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Bamboo contact must not end the episode — fall detected by height
        self.terminations.base_contact = None
