from enum import StrEnum, auto

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api.scenes import Scene
from isaacsim.core.utils.stage import add_reference_to_stage

from src.base.task import Arms, DualDvrkTask, GoalConfig, RigidOffsetPrim, Side
from src.constants import props_dir
from src.plan.plan import Action, ActionType
from src.tasks.schemas import Problem


class Prims(StrEnum):
    NEEDLE = auto()
    RED_RING = auto()
    GREEN_RING = auto()
    BLUE_RING = auto()
    PINK_RING = auto()
    YELLOW_RING = auto()
    GOAL = auto()


class Ring(RigidOffsetPrim):
    def __init__(
        self,
        prim_path,
        name="xform_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
    ):
        contact_points = {
            Side.LEFT_POINT: np.array([0.0, -0.01, 0.02]),
            Side.RIGHT_POINT: np.array([0.0, 0.01, 0.02]),
        }
        approach_points = {
            Side.LEFT_POINT: np.array([0.0, -0.025, 0.02]),
            Side.RIGHT_POINT: np.array([0.0, 0.025, 0.02]),
        }
        departure_points = approach_points
        super().__init__(
            prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            reset_xform_properties=reset_xform_properties,
            contact_points=contact_points,
            approach_points=approach_points,
            departure_points=departure_points,
        )


class Needle(RigidOffsetPrim):
    def __init__(
        self,
        prim_path,
        name="xform_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
    ):
        grasp_points = {
            Side.LEFT_POINT: np.array([0.0, -0.01, -0.001]),
            Side.RIGHT_POINT: np.array([0.0, 0.01, -0.001]),
        }
        approach_points = {
            Side.LEFT_POINT: np.array([0.0, -0.01, 0.03]),
            Side.RIGHT_POINT: np.array([0.0, 0.01, 0.03]),
        }
        departure_points = {
            Side.LEFT_POINT: np.array([0.0, -0.03, 0.005]),
            Side.RIGHT_POINT: np.array([0.0, 0.03, 0.005]),
        }
        super().__init__(
            prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            reset_xform_properties=reset_xform_properties,
            contact_points=grasp_points,
            approach_points=approach_points,
            departure_points=departure_points,
        )


class Goal(RigidOffsetPrim):
    def __init__(
        self,
        prim_path,
        name="xform_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
    ):
        grasp_points = {
            Side.LEFT_POINT: np.array([0.0, -0.01, 0.0]),
            Side.RIGHT_POINT: np.array([0.0, 0.01, 0.0]),
        }
        approach_points = {
            Side.LEFT_POINT: np.array([0.0, 0.0, 0.03]),
            Side.RIGHT_POINT: np.array([0.0, 0.0, 0.03]),
        }
        super().__init__(
            prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            reset_xform_properties=reset_xform_properties,
            contact_points=grasp_points,
            approach_points=approach_points,
        )


# NOTE: If we really feel like it we can also add a variation of this with multiple needles
#       that need to be transferred to different goals in different sequences
class NeedleTransfer(DualDvrkTask):
    def __init__(self, name, problem):
        super().__init__(name, gripper_closed_position=np.array([-0.125, 0.125]))
        self.rings = [
            Prims.RED_RING,
            Prims.BLUE_RING,
            Prims.GREEN_RING,
            Prims.PINK_RING,
            Prims.YELLOW_RING,
        ]
        self.preference = {
            Arms.LEFT_ARM: Side.LEFT_POINT,
            Arms.RIGHT_ARM: Side.RIGHT_POINT,
        }
        self._arm_holding_needle = {
            Arms.LEFT_ARM: False,
            Arms.RIGHT_ARM: False,
        }
        self._last_move_target_by_arm = {
            Arms.LEFT_ARM: None,
            Arms.RIGHT_ARM: None,
        }
        self.passed_through_ring_order: list[str] = []
        self.required_passed_through_ring_order: list[str] | None = None
        self.problem = problem

    def set_up_scene(self, scene: Scene):
        super().set_up_scene(scene)
        z_rot_90 = rot_utils.euler_angles_to_quats(
            np.array([0.0, 0.0, 90.0]), degrees=True
        )
        ring_scale = np.array([0.01, 0.01, 0.01])

        needle_path = props_dir / "needle.usd"
        ring_path = props_dir / "hole_with_stand.usd"
        goal_path = props_dir / "goal.usd"

        add_reference_to_stage(usd_path=str(needle_path), prim_path="/World/Needle")
        self.needle: Needle = scene.add(
            Needle(
                prim_path="/World/Needle",
                name="Needle",
                position=np.array([-0.1, 0.0, 0.0]),
                orientation=z_rot_90,
            )
        )
        add_reference_to_stage(usd_path=str(goal_path), prim_path="/World/Goal")
        self.goal_position: Goal = scene.add(
            Goal(
                prim_path="/World/Goal",
                name="Goal",
                position=np.array([0.1, 0.0, 0.0]),
                orientation=z_rot_90,
            )
        )
        self.goal_position.apply_visual_material(self.black)

        if self.problem == Problem.NEEDLE_TRANSFER_1:
            red_ring_pose = (np.array([0.0, 0.0, 0.0]), z_rot_90)
            green_ring_pose = None
            blue_ring_pose = None
        elif self.problem == Problem.NEEDLE_TRANSFER_2:
            green_ring_pose = (np.array([0.0, -0.025, 0.0]), z_rot_90)
            red_ring_pose = (np.array([0.0, 0.0, 0.0]), z_rot_90)
            blue_ring_pose = (np.array([0.0, 0.025, 0.0]), z_rot_90)

        if red_ring_pose:
            add_reference_to_stage(usd_path=str(ring_path), prim_path="/World/Red_Ring")
            self.red_ring: Ring = scene.add(
                Ring(
                    prim_path="/World/Red_Ring",
                    name="Red_Ring",
                    position=red_ring_pose[0],
                    orientation=red_ring_pose[1],
                    scale=ring_scale,
                )
            )
            self.red_ring.apply_visual_material(self.red)
        if green_ring_pose:
            print("Adding green ring")
            add_reference_to_stage(
                usd_path=str(ring_path), prim_path="/World/Green_Ring"
            )
            self.green_ring: Ring = scene.add(
                Ring(
                    prim_path="/World/Green_Ring",
                    name="Green_Ring",
                    position=green_ring_pose[0],
                    orientation=green_ring_pose[1],
                    scale=ring_scale,
                )
            )
            self.green_ring.apply_visual_material(self.green)
        if blue_ring_pose:
            add_reference_to_stage(
                usd_path=str(ring_path), prim_path="/World/Blue_Ring"
            )
            self.blue_ring: Ring = scene.add(
                Ring(
                    prim_path="/World/Blue_Ring",
                    name="Blue_Ring",
                    position=blue_ring_pose[0],
                    orientation=blue_ring_pose[1],
                    scale=ring_scale,
                )
            )
            self.blue_ring.apply_visual_material(self.blue)

        if self.problem == Problem.NEEDLE_TRANSFER_1:
            self.goal = [
                GoalConfig(
                    self.needle,
                    self.goal_position,
                    passed_through_ring_order=self.passed_through_ring_order,
                    required_passed_through_ring_order=[Prims.RED_RING],
                )
            ]
        elif self.problem == Problem.NEEDLE_TRANSFER_2:
            self.goal = [
                GoalConfig(
                    self.needle,
                    self.goal_position,
                    passed_through_ring_order=self.passed_through_ring_order,
                    required_passed_through_ring_order=[
                        Prims.GREEN_RING,
                        Prims.RED_RING,
                        Prims.BLUE_RING,
                    ],
                )
            ]

    def post_reset(self):
        super().post_reset()
        for arm in self._arm_holding_needle:
            self._arm_holding_needle[arm] = False
            self._last_move_target_by_arm[arm] = None
        self.passed_through_ring_order.clear()

    def _retarget_move_action(
        self, action: Action, side: Side, prim: Prims, last_prim: Prims | None
    ) -> None:
        if last_prim is not None:
            print(f"Adding departure point for {last_prim}")
            # Add departure point of last prim to trajectory
            self.get_prim(last_prim).set_active_side(side)
            action.waypoint_targets.append(
                self.get_prim(last_prim).get_world_departure_pose()
            )
        if prim == Prims.NEEDLE:
            self.needle.set_active_side(side)
            action.target_position, action.target_orientation = (
                self.needle.get_world_pose_with_offset()
            )
            action.waypoint_targets.append(self.needle.get_world_approach_pose())
        elif prim in self.rings:
            ring_prim = self.get_prim(prim)
            ring_prim.set_active_side(side)
            action.target_position, action.target_orientation = (
                ring_prim.get_world_pose_with_offset()
            )
            action.waypoint_targets.append(ring_prim.get_world_approach_pose())
        else:
            # Handling goal position
            self.goal_position.set_active_side(side)
            action.target_position, action.target_orientation = (
                self.goal_position.get_world_pose_with_offset()
            )
            action.waypoint_targets.append(self.goal_position.get_world_approach_pose())
        action.waypoint_reached = False

    def prepare_action(self, action: Action, arm: Arms, ee_pos: np.ndarray):
        if action.action_type == ActionType.MOVE:
            selected_side = self.preference[arm]
            self._retarget_move_action(
                action,
                selected_side,
                action.target_prim_name,
                self._last_move_target_by_arm[arm],
            )
            self._last_move_target_by_arm[arm] = action.target_prim_name
            if self._arm_holding_needle[arm] and action.target_prim_name in self.rings:
                # Moved needle through ring, append to 'threaded' rings
                self.passed_through_ring_order.append(action.target_prim_name)
        elif action.action_type == ActionType.PICK:
            if self._last_move_target_by_arm[arm] == Prims.NEEDLE:
                self._arm_holding_needle[arm] = True
        elif action.action_type == ActionType.PLACE:
            if self._arm_holding_needle[arm]:
                self._arm_holding_needle[arm] = False

    def get_prim(self, prim_name: Prims) -> RigidOffsetPrim | None:
        match prim_name:
            case Prims.NEEDLE:
                return self.needle
            case Prims.RED_RING:
                return self.red_ring
            case Prims.GREEN_RING:
                return self.green_ring
            case Prims.BLUE_RING:
                return self.blue_ring
            case Prims.GOAL:
                return self.goal_position
            case _:
                return None
