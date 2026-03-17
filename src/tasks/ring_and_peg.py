import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.materials import OmniPBR
from isaacsim.core.prims import SingleXFormPrim, XFormPrim

from src.base.prim import RigidOffsetPrim
from src.base.task import DvrkTask, GoalConfig
from src.constants import props_dir
from src.tasks.schemas import Problem


# TODO: Additionally create a variant of this with more possible pegs and rings
# TODO: Either figure out how to repeatabilly allow for stacking the rings or only allow for
#       problems with single-capacity pegs


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
        super().__init__(
            prim_path,
            np.array([0.01, 0.0, -0.004]),
            name,
            position,
            translation,
            orientation,
            scale,
            visible,
            reset_xform_properties,
        )


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
        super().__init__(
            prim_path,
            np.array([0.01, 0.0, 0.0]),
            name,
            position,
            translation,
            orientation,
            scale,
            visible,
            reset_xform_properties,
        )


class RingAndPeg(DvrkTask):
    def __init__(self, name, problem):
        super().__init__(name, gripper_closed_position=np.array([-0.2, 0.2]))
        self.peg_xy_position = {
            "Peg": [0.0, 0.0],
            "Peg_01": [0.03, 0.0],
            "Peg_02": [0.0, 0.03],
            "Peg_03": [0.0, -0.03],
            "Peg_04": [-0.03, 0.0],
        }

        self.problem = problem

        self.red = OmniPBR(
            prim_path="/World/Looks/Red",
            name="Red",
            color=np.array([1.0, 0.0, 0.0]),
        )
        self.green = OmniPBR(
            prim_path="/World/Looks/Green",
            name="Green",
            color=np.array([0.0, 1.0, 0.0]),
        )
        self.blue = OmniPBR(
            prim_path="/World/Looks/Blue",
            name="Blue",
            color=np.array([0.0, 0.0, 1.0]),
        )
        self.pink = OmniPBR(
            prim_path="/World/Looks/Pink", name="Pink", color=np.array([1.0, 0.0, 1.0])
        )
        self.yellow = OmniPBR(
            prim_path="/World/Looks/Yellow",
            name="Yellow",
            color=np.array([1.0, 1.0, 0.0]),
        )

    def set_up_scene(self, scene):
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

        red_peg_name = "Peg"
        green_peg_name = "Peg_01"
        blue_peg_name = "Peg_03"
        pink_peg_name = "Peg_02"
        yellow_peg_name = "Peg_04"

        self.red_peg = Peg(f"/World/Pegs/pegs/{red_peg_name}", "red_peg_view")
        self.green_peg = Peg(f"/World/Pegs/pegs/{green_peg_name}", "green_peg_view")
        self.blue_peg = Peg(f"/World/Pegs/pegs/{blue_peg_name}", "blue_peg_view")
        self.pink_peg = Peg(f"/World/Pegs/pegs/{pink_peg_name}", "pink_peg_view")
        self.yellow_peg = Peg(f"/World/Pegs/pegs/{yellow_peg_name}", "yellow_peg_view")

        self.red_peg.apply_visual_material(self.red)
        self.green_peg.apply_visual_material(self.green)
        self.blue_peg.apply_visual_material(self.blue)
        self.pink_peg.apply_visual_material(self.pink)
        self.yellow_peg.apply_visual_material(self.yellow)

        if self.problem == Problem.RING_AND_PEG_1:
            red_ring_starting_peg = pink_peg_name
            green_ring_starting_peg = yellow_peg_name
            blue_ring_starting_peg = red_peg_name
            yellow_ring_starting_peg = None
            pink_ring_starting_peg = None
        elif self.problem == Problem.RING_AND_PEG_2:
            red_ring_starting_peg = None
            green_ring_starting_peg = None
            blue_ring_starting_peg = red_peg_name
            yellow_ring_starting_peg = blue_peg_name
            pink_ring_starting_peg = green_peg_name
        elif self.problem == Problem.RING_AND_PEG_3:
            red_ring_starting_peg = red_peg_name
            green_ring_starting_peg = yellow_peg_name
            blue_ring_starting_peg = None
            yellow_ring_starting_peg = blue_peg_name
            pink_ring_starting_peg = green_peg_name
        elif self.problem == Problem.RING_AND_PEG_5:
            red_ring_starting_peg = pink_peg_name
            green_ring_starting_peg = green_peg_name
            blue_ring_starting_peg = blue_peg_name
            yellow_ring_starting_peg = red_peg_name
            pink_ring_starting_peg = None

        if red_ring_starting_peg:
            add_reference_to_stage(usd_path=str(ring_path), prim_path="/World/Red_Ring")
            self.red_ring = scene.add(
                Ring(
                    prim_path="/World/Red_Ring",
                    name="Red_Ring",
                    position=np.array(
                        [*self.peg_xy_position[red_ring_starting_peg], ring_z_offset]
                    ),
                )
            )
            self.red_ring.apply_visual_material(self.red)
        if green_ring_starting_peg:
            add_reference_to_stage(
                usd_path=str(ring_path), prim_path="/World/Green_Ring"
            )
            self.green_ring = scene.add(
                Ring(
                    prim_path="/World/Green_Ring",
                    name="Green_Ring",
                    position=np.array(
                        [*self.peg_xy_position[green_ring_starting_peg], ring_z_offset]
                    ),
                )
            )
            self.green_ring.apply_visual_material(self.green)
        if blue_ring_starting_peg:
            add_reference_to_stage(
                usd_path=str(ring_path), prim_path="/World/Blue_Ring"
            )
            self.blue_ring = scene.add(
                Ring(
                    prim_path="/World/Blue_Ring",
                    name="Blue_Ring",
                    position=np.array(
                        [*self.peg_xy_position[blue_ring_starting_peg], ring_z_offset]
                    ),
                )
            )
            self.blue_ring.apply_visual_material(self.blue)
        if yellow_ring_starting_peg:
            add_reference_to_stage(
                usd_path=str(ring_path), prim_path="/World/Yellow_Ring"
            )
            self.yellow_ring = scene.add(
                Ring(
                    prim_path="/World/Yellow_Ring",
                    name="Yellow_Ring",
                    position=np.array(
                        [*self.peg_xy_position[yellow_ring_starting_peg], ring_z_offset]
                    ),
                )
            )
            self.yellow_ring.apply_visual_material(self.yellow)
        if pink_ring_starting_peg:
            add_reference_to_stage(
                usd_path=str(ring_path), prim_path="/World/Pink_Ring"
            )
            self.pink_ring = scene.add(
                Ring(
                    prim_path="/World/Pink_Ring",
                    name="Pink_Ring",
                    position=np.array(
                        [*self.peg_xy_position[pink_ring_starting_peg], ring_z_offset]
                    ),
                )
            )
            self.pink_ring.apply_visual_material(self.pink)

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
        elif self.problem == Problem.RING_AND_PEG_5:
            self.goal = [
                GoalConfig(self.red_ring, self.red_peg),
                GoalConfig(self.green_ring, self.green_peg),
                GoalConfig(self.blue_ring, self.blue_peg),
                GoalConfig(self.yellow_ring, self.yellow_peg),
            ]

    def get_prim(self, prim_name: str) -> SingleXFormPrim | None:
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
