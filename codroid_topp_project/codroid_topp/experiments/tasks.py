from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from codroid_topp.robot.virtual_srs import VirtualSRSModel
from codroid_topp.trajectory.se3_path import Pose


@dataclass(frozen=True)
class CupTransferSceneConfig:
    """World-frame scene constants shared by the task and MeshCat animation.

    The robot URDF base frame is kept unchanged. All positions below are expressed in
    the same robot world frame, so the Cartesian task, the tabletop scene, and the
    animation are geometrically consistent.
    """

    floor_z_world: float
    floor_size_xyz: tuple[float, float, float]
    floor_center_xy: tuple[float, float]
    wall_back_y: float
    wall_left_x: float
    wall_right_x: float
    wall_height: float
    table_center_xy: tuple[float, float]
    table_top_z_world: float
    table_top_size_xyz: tuple[float, float, float]
    cup_radius_m: float
    cup_height_m: float
    top_grasp_clearance_m: float
    cup_pick_center_world: np.ndarray
    cup_place_center_world: np.ndarray
    T_ee_cup: np.ndarray
    grasp_s: float
    place_s: float
    contact_threshold_m: float = 0.035

    def cup_pick_T_world(self) -> np.ndarray:
        return _make_T(np.eye(3), self.cup_pick_center_world)

    def cup_place_T_world(self) -> np.ndarray:
        return _make_T(np.eye(3), self.cup_place_center_world)

    def cup_pick_top_world(self) -> np.ndarray:
        top = np.asarray(self.cup_pick_center_world, dtype=float).reshape(3).copy()
        top[2] += 0.5 * float(self.cup_height_m)
        return top

    def cup_place_top_world(self) -> np.ndarray:
        top = np.asarray(self.cup_place_center_world, dtype=float).reshape(3).copy()
        top[2] += 0.5 * float(self.cup_height_m)
        return top

    def grasp_target_world(self) -> np.ndarray:
        target = self.cup_pick_top_world().copy()
        target[2] += float(self.top_grasp_clearance_m)
        return target

    def place_target_world(self) -> np.ndarray:
        target = self.cup_place_top_world().copy()
        target[2] += float(self.top_grasp_clearance_m)
        return target


@dataclass
class HouseholdTask:
    name: str
    keyframes: list[Pose]
    q_waypoints: np.ndarray
    description: str
    scene_config: CupTransferSceneConfig | None = None
    variant_id: int = 0
    seed: int = 0
    perturbation: dict[str, float] | None = None


# A reachable, publication-grade grasp orientation for the left arm.
# Columns are the x/y/z axes of link_arm_l_07 expressed in the URDF world frame.
# It is kept fixed over the task so the end-effector motion stays smooth and the cup
# remains upright in the world frame once attached.
CUP_GRASP_R_WORLD_FROM_EE = np.array(
    [
        [-0.2948712583, 0.5797155651, 0.7595925254],
        [-0.9154067967, -0.3993388610, -0.0505852799],
        [0.2740097398, -0.7102523056, 0.6484291209],
    ],
    dtype=float,
)


def _make_T(R: np.ndarray, p: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T

def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert a small rotation vector to a 3x3 rotation matrix.

    Used by task perturbations to add small reproducible orientation changes.
    """
    rotvec = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=float)

    axis = rotvec / theta
    x, y, z = axis
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

# Use a top-side grasp proxy rather than a side-center grasp. The cup center is kept
# directly below the tool origin in the WORLD vertical direction so the end-effector
# visually stays above the cup and no longer appears to punch through the tabletop.
# The chosen clearance keeps the gripper above the rim while remaining close enough
# for a convincing pick.
_TOP_GRASP_CLEARANCE_M = 0.050
_CUP_TO_EE_WORLD_OFFSET = np.array([0.0, 0.0, -(0.060 + _TOP_GRASP_CLEARANCE_M)], dtype=float)
T_EE_CUP_TOP_GRASP = np.eye(4, dtype=float)
T_EE_CUP_TOP_GRASP[:3, :3] = CUP_GRASP_R_WORLD_FROM_EE.T
T_EE_CUP_TOP_GRASP[:3, 3] = CUP_GRASP_R_WORLD_FROM_EE.T @ _CUP_TO_EE_WORLD_OFFSET

# Backward-compatible alias retained for existing imports.
T_EE_CUP_SIDE_GRASP = T_EE_CUP_TOP_GRASP


def _pose_from_R_p(
    R: np.ndarray,
    p: np.ndarray | list[float] | tuple[float, float, float],
) -> Pose:
    return Pose.from_T(_make_T(R, p))


def _ee_position_for_cup_center(cup_center_world: np.ndarray) -> np.ndarray:
    """Return the link_arm_l_07 origin that places the grasped cup at cup_center_world."""
    return (
        np.asarray(cup_center_world, dtype=float).reshape(3)
        - CUP_GRASP_R_WORLD_FROM_EE @ T_EE_CUP_TOP_GRASP[:3, 3]
    )


def cup_transfer_scene_config() -> CupTransferSceneConfig:
    """Scene and pick/place geometry for a front-facing tabletop transfer.

    Design goals compared with the original demo:
    1) The table and cup are moved to the robot's FRONT instead of its lateral side.
    2) The room layout better matches a polished household-service-robot paper figure.
    3) The Cartesian pick/place targets are defined above the cup, avoiding visible
       table penetration during the grasp phase.
    """

    cup_height = 0.120
    table_top_z = -0.430
    cup_center_z = table_top_z + 0.5 * cup_height

    # Put the interaction directly in front of the robot. The left arm still gets a
    # mild positive-y bias for reachability, but the tabletop is no longer off to the
    # robot's side.
    pick_center = np.array([0.340, 0.100, cup_center_z], dtype=float)
    place_center = np.array([0.460, 0.140, cup_center_z], dtype=float)

    return CupTransferSceneConfig(
        floor_z_world=-1.050,
        floor_size_xyz=(2.45, 2.00, 0.035),
        floor_center_xy=(0.35, 0.10),
        wall_back_y=0.98,
        wall_left_x=-0.78,
        wall_right_x=1.26,
        wall_height=1.72,
        table_center_xy=(0.400, 0.120),
        table_top_z_world=table_top_z,
        table_top_size_xyz=(0.86, 0.62, 0.050),
        cup_radius_m=0.038,
        cup_height_m=cup_height,
        top_grasp_clearance_m=_TOP_GRASP_CLEARANCE_M,
        cup_pick_center_world=pick_center,
        cup_place_center_world=place_center,
        T_ee_cup=T_EE_CUP_TOP_GRASP.copy(),
        grasp_s=3.0 / 7.0,
        place_s=6.0 / 7.0,
        contact_threshold_m=0.035,
    )


def get_household_scene_config(name: str = "cup_transfer") -> CupTransferSceneConfig:
    if name != "cup_transfer":
        # The compact tabletop room also provides a coherent default scene for the
        # remaining demonstrations.
        return cup_transfer_scene_config()
    return cup_transfer_scene_config()


def _seed_waypoints_deg() -> np.ndarray:
    """IK seed hints for the explicit Cartesian cup-transfer poses.

    These are not used to define the path. They only provide a repeatable 7-DoF seed
    close to the first Cartesian keyframe.
    """

    q_deg = np.array(
        [
            [-70.004641, -66.010790, 109.323263, -111.491007, -21.996931, 13.492228, 56.157536],
            [-53.572085, -58.591945, 89.807556, -89.112116, -9.303724, -0.746201, 41.770549],
            [-41.367569, -55.594292, 73.216772, -74.973247, 1.819688, -18.127764, 28.137750],
            [-39.993918, -68.138213, 73.626981, -66.705957, 8.009523, -17.221568, 19.851266],
            [-39.124240, -44.155062, 86.903937, -84.900611, -6.152505, -23.918562, 26.867756],
            [-52.576888, -34.693809, 76.943889, -92.084301, -18.311533, -31.278945, 25.410332],
            [-69.298996, -34.116458, 82.630366, -83.553788, -29.930821, -32.384739, 15.399707],
            [-85.596206, -75.000514, 108.625546, -69.776289, -40.980224, -10.614995, 26.761818],
        ],
        dtype=float,
    )
    return np.deg2rad(q_deg)


def _make_cup_transfer_task() -> HouseholdTask:
    scene = cup_transfer_scene_config()
    R = CUP_GRASP_R_WORLD_FROM_EE

    pick_ee = _ee_position_for_cup_center(scene.cup_pick_center_world)
    place_ee = _ee_position_for_cup_center(scene.cup_place_center_world)

    key_positions = [
        # 0. Ready pose in front of the torso and clear of the table.
        np.array([0.220, 0.020, -0.180], dtype=float),
        # 1. Long-range approach toward the front tabletop.
        pick_ee + np.array([-0.060, 0.000, 0.100], dtype=float),
        # 2. Hover above the mug.
        pick_ee + np.array([0.000, 0.000, 0.040], dtype=float),
        # 3. Top-side grasp target: tool origin stays above the cup.
        pick_ee,
        # 4. Lift vertically before translating.
        pick_ee + np.array([0.000, 0.000, 0.160], dtype=float),
        # 5. Translate while remaining above the placement location.
        place_ee + np.array([0.000, 0.000, 0.160], dtype=float),
        # 6. Lower for placement, still without penetrating the tabletop.
        place_ee,
        # 7. Retreat upward and slightly back toward the robot.
        place_ee + np.array([-0.040, -0.080, 0.140], dtype=float),
    ]

    keyframes = [_pose_from_R_p(R, p) for p in key_positions]

    return HouseholdTask(
        name="cup_transfer",
        keyframes=keyframes,
        q_waypoints=_seed_waypoints_deg(),
        description=(
            "Front-facing Cartesian cup transfer with a polished tabletop scene: "
            "approach from above, top-side grasp, lift, transfer, place, and retreat."
        ),
        scene_config=scene,
    )


def _simple_cartesian_task(
    name: str,
    positions: list[list[float]],
    description: str,
) -> HouseholdTask:
    R = CUP_GRASP_R_WORLD_FROM_EE
    keyframes = [_pose_from_R_p(R, p) for p in positions]

    seed = _seed_waypoints_deg()[0]
    q_waypoints = np.tile(seed.reshape(1, 7), (len(keyframes), 1))

    return HouseholdTask(
        name=name,
        keyframes=keyframes,
        q_waypoints=q_waypoints,
        description=description,
        scene_config=get_household_scene_config("cup_transfer"),
    )


def make_household_task(model: VirtualSRSModel, name: str = "cup_transfer") -> HouseholdTask:
    """Generate repeatable Cartesian keyframes for household demos.

    The geometric task is specified directly in Cartesian SE(3) poses. The returned
    q_waypoints are retained only as IK seed hints for existing planner code that
    expects an initial 7-DoF vector.
    """

    # Keep the model argument for API compatibility with older scripts. The cup path
    # is no longer generated by model.fk(q_waypoint).
    _ = model

    if name == "cup_transfer":
        return _make_cup_transfer_task()

    if name == "drawer_reach":
        return _simple_cartesian_task(
            name="drawer_reach",
            positions=[
                [0.230, 0.330, -0.230],
                [0.120, 0.450, -0.330],
                [0.080, 0.560, -0.340],
                [0.200, 0.390, -0.250],
            ],
            description="Cartesian reach toward a drawer handle and retract.",
        )

    if name == "medicine_handover":
        return _simple_cartesian_task(
            name="medicine_handover",
            positions=[
                [0.220, 0.340, -0.240],
                [0.070, 0.420, -0.370],
                [-0.050, 0.500, -0.300],
                [-0.180, 0.470, -0.220],
            ],
            description="Cartesian transfer of a medicine box from table to user side.",
        )

    if name == "fast_lift_transfer":
        return _simple_cartesian_task(
            name="fast_lift_transfer",
            positions=[
                [0.240, 0.120, -0.360],
                [0.360, 0.160, -0.460],
                [0.430, 0.180, -0.260],
                [0.500, 0.150, -0.080],
                [0.360, 0.050, -0.150],
                [0.250, 0.000, -0.300],
            ],
            description=(
                "Aggressive vertical lift-and-transfer task designed to activate shoulder/elbow "
                "acceleration and torque constraints more strongly than the tabletop tasks."
            ),
        )

    if name == "near_limit_reach":
        return _simple_cartesian_task(
            name="near_limit_reach",
            positions=[
                [0.180, 0.160, -0.240],
                [0.050, 0.410, -0.340],
                [-0.080, 0.570, -0.270],
                [-0.160, 0.610, -0.180],
                [-0.060, 0.480, -0.260],
            ],
            description=(
                "Near-limit reach task that pushes the redundant left arm toward low-margin "
                "configurations and makes IK-layer psi selection more visible."
            ),
        )

    raise ValueError(f"Unknown task {name!r}")


def _task_with_perturbation(task: HouseholdTask, rng: np.random.Generator, scale: float) -> HouseholdTask:
    if scale <= 0.0:
        return task
    keyframes = []
    max_xyz = np.array([0.025, 0.025, 0.020], dtype=float) * float(scale)
    rot_scale = 0.035 * float(scale)
    for i, pose in enumerate(task.keyframes):
        # Keep the first keyframe nearly fixed so IK seeding remains stable.
        gain = 0.25 if i == 0 else 1.0
        dp = rng.uniform(-max_xyz, max_xyz) * gain
        if task.name == "fast_lift_transfer":
            dp[2] += rng.uniform(-0.018, 0.028) * gain
        if task.name == "near_limit_reach" and i in (2, 3):
            dp += np.array([-0.015, 0.020, 0.010]) * float(scale)
        rotvec = rng.normal(0.0, rot_scale * gain, size=3)
        R = _rotvec_to_matrix(rotvec) @ pose.R
        keyframes.append(Pose(pose.p + dp, R))
    q_wp = np.asarray(task.q_waypoints, dtype=float).copy()
    if q_wp.size:
        q_wp = q_wp + rng.normal(0.0, 0.012 * float(scale), size=q_wp.shape)
    return HouseholdTask(
        name=task.name,
        keyframes=keyframes,
        q_waypoints=q_wp,
        description=task.description + " (seeded perturbation variant)",
        scene_config=task.scene_config,
        variant_id=task.variant_id,
        seed=task.seed,
        perturbation={"position_scale_m": float(np.max(max_xyz)), "rotation_scale_rad": float(rot_scale)},
    )


def make_household_task_variant(
    model: VirtualSRSModel,
    name: str = "cup_transfer",
    variant_id: int = 0,
    seed: int = 42,
    perturb_scale: float = 0.0,
) -> HouseholdTask:
    """Generate deterministic task variants for paper statistics.

    The base path remains interpretable, while future keyframes receive small
    reproducible perturbations. This makes mean/std tables meaningful without
    changing the robot model or requiring online perception.
    """
    task = make_household_task(model, name)
    object.__setattr__(task, "variant_id", int(variant_id)) if hasattr(task, "__setattr__") else None
    task.variant_id = int(variant_id)
    task.seed = int(seed)
    stable_name_hash = zlib.crc32(name.encode("utf-8")) % 997
    rng = np.random.default_rng(int(seed) + 1009 * int(variant_id) + stable_name_hash)
    out = _task_with_perturbation(task, rng, float(perturb_scale))
    out.variant_id = int(variant_id)
    out.seed = int(seed)
    if out.perturbation is None:
        out.perturbation = {"position_scale_m": 0.0, "rotation_scale_rad": 0.0}
    out.perturbation.update({"variant_id": float(variant_id), "seed": float(seed), "perturb_scale": float(perturb_scale)})
    return out
