from enum import StrEnum, auto

import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.scenes import Scene
from isaacsim.core.prims import XFormPrim

from src.base.prim import RigidOffsetPrim
from src.base.task import GoalConfig, SingleDvrkTask
from src.plan.plan import Action, ActionType
from src.constants import props_dir
from src.tasks.schemas import Problem


class Prims(StrEnum):
    RED_RING = auto()
    GREEN_RING = auto()
    BLUE_RING = auto()
    PINK_RING = auto()
    YELLOW_RING = auto()
    RED_PEG = "Peg"
    GREEN_PEG = "Peg_01"
    PINK_PEG = "Peg_02"
    BLUE_PEG = "Peg_03"
    YELLOW_PEG = "Peg_04"


class Peg(RigidOffsetPrim):
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
        peg_points = {
            "default": np.array([0.01, 0.0, -0.0038]),
        }
        peg_approach_points = {
            "default": np.array([0.015, 0.0, 0.02]),
        }
        peg_departure_points = peg_approach_points
        super().__init__(
            prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            reset_xform_properties=reset_xform_properties,
            contact_points=peg_points,
            approach_points=peg_approach_points,
            departure_points=peg_departure_points,
            # Referencing the world frame delivers more stable results for this task
            offset_in_local_frame=False,
        )
        # Make sure contact point is initialized correctly
        self.set_active_side("default")


class Ring(RigidOffsetPrim): ...


class RingAndPeg(SingleDvrkTask):
    def __init__(self, name, problem):
        super().__init__(name, gripper_closed_position=np.array([-0.19, 0.19]))
        self.peg_xy_position = {
            Prims.RED_PEG: [0.0, 0.0],
            Prims.GREEN_PEG: [0.03, 0.0],
            Prims.PINK_PEG: [0.0, 0.03],
            Prims.BLUE_PEG: [0.0, -0.03],
            Prims.YELLOW_PEG: [-0.03, 0.0],
        }
        self.problem = problem
        self.last_move_target = None
        self.gripping = False

    def _add_ring(
        self,
        scene: Scene,
        ring_path: str,
        prim_path: str,
        name: str,
        position: np.ndarray,
        material,
    ) -> Ring:
        add_reference_to_stage(usd_path=ring_path, prim_path=prim_path)
        ring = scene.add(
            Ring(
                prim_path=prim_path,
                name=name,
                position=position,
            )
        )
        ring.apply_visual_material(material)
        return ring

    def set_up_scene(self, scene: Scene):
        super().set_up_scene(scene)
        pegs_path = props_dir / "pegs.usd"
        ring_path = props_dir / "ring.usd"
        add_reference_to_stage(usd_path=str(pegs_path), prim_path="/World/Pegs")
        self.pegs = scene.add(
            XFormPrim(
                prim_paths_expr="/World/Pegs",
                name="Pegs",
            )
        )
        ring_z_offset = 0.0011

        self.red_peg = Peg(f"/World/Pegs/pegs/{Prims.RED_PEG}", "red_peg_view")
        self.green_peg = Peg(f"/World/Pegs/pegs/{Prims.GREEN_PEG}", "green_peg_view")
        self.blue_peg = Peg(f"/World/Pegs/pegs/{Prims.BLUE_PEG}", "blue_peg_view")
        self.pink_peg = Peg(f"/World/Pegs/pegs/{Prims.PINK_PEG}", "pink_peg_view")
        self.yellow_peg = Peg(f"/World/Pegs/pegs/{Prims.YELLOW_PEG}", "yellow_peg_view")

        self.red_peg.apply_visual_material(self.red)
        self.green_peg.apply_visual_material(self.green)
        self.blue_peg.apply_visual_material(self.blue)
        self.pink_peg.apply_visual_material(self.pink)
        self.yellow_peg.apply_visual_material(self.yellow)

        if self.problem == Problem.RING_AND_PEG_1:
            red_ring_peg = Prims.PINK_PEG
            green_ring_peg = Prims.YELLOW_PEG
            blue_ring_peg = Prims.RED_PEG
            yellow_ring_peg = None
            pink_ring_peg = None
        elif self.problem == Problem.RING_AND_PEG_2:
            red_ring_peg = None
            green_ring_peg = None
            blue_ring_peg = Prims.RED_PEG
            yellow_ring_peg = Prims.BLUE_PEG
            pink_ring_peg = Prims.GREEN_PEG
        elif self.problem == Problem.RING_AND_PEG_3:
            red_ring_peg = Prims.RED_PEG
            green_ring_peg = Prims.YELLOW_PEG
            blue_ring_peg = None
            yellow_ring_peg = Prims.BLUE_PEG
            pink_ring_peg = Prims.GREEN_PEG
        elif self.problem == Problem.RING_AND_PEG_4:
            red_ring_peg = Prims.PINK_PEG
            green_ring_peg = Prims.GREEN_PEG
            blue_ring_peg = Prims.BLUE_PEG
            yellow_ring_peg = Prims.RED_PEG
            pink_ring_peg = None
        elif self.problem == Problem.RING_AND_PEG_5:
            red_ring_peg = None
            green_ring_peg = Prims.BLUE_PEG
            blue_ring_peg = Prims.PINK_PEG
            yellow_ring_peg = Prims.GREEN_PEG
            pink_ring_peg = Prims.YELLOW_PEG

        self.rings: list[RigidOffsetPrim] = []
        if red_ring_peg:
            self.red_ring = self._add_ring(
                scene=scene,
                ring_path=str(ring_path),
                prim_path="/World/Red_Ring",
                name="Red_Ring",
                position=np.array([*self.peg_xy_position[red_ring_peg], ring_z_offset]),
                material=self.red,
            )
            self.rings.append(self.red_ring)
        if green_ring_peg:
            self.green_ring = self._add_ring(
                scene=scene,
                ring_path=str(ring_path),
                prim_path="/World/Green_Ring",
                name="Green_Ring",
                position=np.array(
                    [*self.peg_xy_position[green_ring_peg], ring_z_offset]
                ),
                material=self.green,
            )
            self.rings.append(self.green_ring)
        if blue_ring_peg:
            self.blue_ring = self._add_ring(
                scene=scene,
                ring_path=str(ring_path),
                prim_path="/World/Blue_Ring",
                name="Blue_Ring",
                position=np.array(
                    [*self.peg_xy_position[blue_ring_peg], ring_z_offset]
                ),
                material=self.blue,
            )
            self.rings.append(self.blue_ring)
        if yellow_ring_peg:
            self.yellow_ring = self._add_ring(
                scene=scene,
                ring_path=str(ring_path),
                prim_path="/World/Yellow_Ring",
                name="Yellow_Ring",
                position=np.array(
                    [*self.peg_xy_position[yellow_ring_peg], ring_z_offset]
                ),
                material=self.yellow,
            )
            self.rings.append(self.yellow_ring)
        if pink_ring_peg:
            self.pink_ring = self._add_ring(
                scene=scene,
                ring_path=str(ring_path),
                prim_path="/World/Pink_Ring",
                name="Pink_Ring",
                position=np.array(
                    [*self.peg_xy_position[pink_ring_peg], ring_z_offset]
                ),
                material=self.pink,
            )
            self.rings.append(self.pink_ring)

        if self.problem == Problem.RING_AND_PEG_1:
            self.goal = [
                GoalConfig(self.red_ring, self.red_peg),
                GoalConfig(self.green_ring, self.green_peg),
                GoalConfig(self.blue_ring, self.blue_peg),
            ]
        elif self.problem == Problem.RING_AND_PEG_2:
            self.goal = [
                GoalConfig(self.blue_ring, self.blue_peg),
                GoalConfig(self.yellow_ring, self.yellow_peg),
                GoalConfig(self.pink_ring, self.pink_peg),
            ]
        elif self.problem == Problem.RING_AND_PEG_3:
            self.goal = [
                GoalConfig(self.red_ring, self.red_peg),
                GoalConfig(self.green_ring, self.green_peg),
                GoalConfig(self.yellow_ring, self.yellow_peg),
                GoalConfig(self.pink_ring, self.pink_peg),
            ]
        elif self.problem == Problem.RING_AND_PEG_4:
            self.goal = [
                GoalConfig(self.red_ring, self.red_peg),
                GoalConfig(self.green_ring, self.green_peg),
                GoalConfig(self.blue_ring, self.blue_peg),
                GoalConfig(self.yellow_ring, self.yellow_peg),
            ]
        elif self.problem == Problem.RING_AND_PEG_5:
            self.goal = [
                GoalConfig(self.green_ring, self.green_peg),
                GoalConfig(self.blue_ring, self.blue_peg),
                GoalConfig(self.yellow_ring, self.yellow_peg),
                GoalConfig(self.pink_ring, self.pink_peg),
            ]

    def post_reset(self):
        super().post_reset()
        self.last_move_target = None
        self.gripping = False

    def prepare_action(self, action: Action, robot_name: str, ee_pos: np.ndarray):
        if action.action_type == ActionType.MOVE:
            target_prim = self.get_prim(action.target_prim_name)
            action.target_position, action.target_orientation = (
                target_prim.get_world_pose_with_offset()
            )
            # Check if current move target is occupied
            occupied = False
            target_pos, _ = target_prim.get_world_pose()
            for ring in self.rings:
                ring_pos, _ = ring.get_world_pose()
                dist = np.sqrt(np.sum((ring_pos - target_pos) ** 2))
                if dist <= 0.005:
                    occupied = True
            if self.gripping and occupied:
                # Avoid collisiong with below ring on peg visit in suboptimal plan
                action.target_position = action.target_position + np.array(
                    [0.0, 0.0, 0.005]
                )
            if self.last_move_target is not None:
                # Add departure point of last prim
                action.waypoint_targets.append(
                    self.get_prim(self.last_move_target).get_world_departure_pose()
                )
            action.waypoint_targets.append(target_prim.get_world_approach_pose())
            self.last_move_target = action.target_prim_name
            action.waypoint_reached = False
        if action.action_type == ActionType.PICK:
            self.gripping = True
        else:
            self.gripping = False

    def get_prim(self, prim_name: str) -> RigidOffsetPrim | None:
        match prim_name:
            case "red_peg":
                return self.red_peg
            case "green_peg":
                return self.green_peg
            case "blue_peg":
                return self.blue_peg
            case "pink_peg":
                return self.pink_peg
            case "yellow_peg":
                return self.yellow_peg
            case _:
                return None
