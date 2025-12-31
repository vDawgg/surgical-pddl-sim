import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.materials import OmniPBR
from isaacsim.core.prims import SingleXFormPrim, XFormPrim

from src.base.dvrk_task import DvrkTask
from src.constants import props_dir


class OffsetWrapper(SingleXFormPrim):
    """
    Wrapper for Ring and Peg prims to allow for the definition of an x-offset.
    This offset ensures the EE is able to grasp the Ring at its border and drop it of on the Peg.
    """

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
            name,
            position,
            translation,
            orientation,
            scale,
            visible,
            reset_xform_properties,
        )
        self.offset = np.array([0.01, 0.0, 0.0])

    def get_world_pose(self):
        position, orientation = super().get_world_pose()
        return position + self.offset, orientation


class PegAndRing(DvrkTask):
    # TODO: This should have the option for setting the appropriate problem in this domain
    def __init__(self, name):
        super().__init__(name, gripper_closed_position=np.array([-0.2, 0.2]))
        self.peg_xy_position = {
            "Peg": [0.0, 0.0],
            "Peg_01": [0.03, 0.0],
            "Peg_02": [0.0, 0.03],
            "Peg_03": [0.0, -0.03],
            "Peg_04": [-0.03, 0.0],
        }

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

    # TODO: Make this configurable according to the problem specified in init
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        print("Calling scene setup")
        pegs_path = props_dir / "pegs_no_rigid_body.usd"
        ring_path = props_dir / "ring.usd"
        add_reference_to_stage(usd_path=str(pegs_path), prim_path="/World/Pegs")
        add_reference_to_stage(usd_path=str(ring_path), prim_path="/World/Red_Ring")
        add_reference_to_stage(usd_path=str(ring_path), prim_path="/World/Green_Ring")
        add_reference_to_stage(usd_path=str(ring_path), prim_path="/World/Blue_Ring")
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

        red_ring_starting_peg = "Peg_02"
        green_ring_starting_peg = "Peg_04"
        blue_ring_starting_peg = "Peg"

        self.red_peg = OffsetWrapper(f"/World/Pegs/pegs/{red_peg_name}", "red_peg_view")
        self.green_peg = OffsetWrapper(
            f"/World/Pegs/pegs/{green_peg_name}", "red_peg_view"
        )
        self.blue_peg = OffsetWrapper(
            f"/World/Pegs/pegs/{blue_peg_name}", "blue_peg_view"
        )

        self.red_peg.apply_visual_material(self.red)
        self.green_peg.apply_visual_material(self.green)
        self.blue_peg.apply_visual_material(self.blue)

        self.red_ring = scene.add(
            OffsetWrapper(
                prim_path="/World/Red_Ring",
                name="Red_Ring",
                position=np.array(
                    [*self.peg_xy_position[red_ring_starting_peg], ring_z_offset]
                ),
            )
        )
        self.green_ring = scene.add(
            OffsetWrapper(
                prim_path="/World/Green_Ring",
                name="Green_Ring",
                position=np.array(
                    [*self.peg_xy_position[green_ring_starting_peg], ring_z_offset]
                ),
            )
        )
        self.blue_ring = scene.add(
            OffsetWrapper(
                prim_path="/World/Blue_Ring",
                name="Blue_Ring",
                position=np.array(
                    [*self.peg_xy_position[blue_ring_starting_peg], ring_z_offset]
                ),
            )
        )

        self.red_ring.apply_visual_material(self.red)
        self.green_ring.apply_visual_material(self.green)
        self.blue_ring.apply_visual_material(self.blue)
