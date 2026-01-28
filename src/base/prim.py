import numpy as np
from isaacsim.core.prims import SingleRigidPrim


class RigidOffsetPrim(SingleRigidPrim):
    """
    Wrapper around rigid prims to allow for the definition of an offset to the world position.
    This ensures proper placement at grippable positions of the object.
    Use this for physics-enabled objects that move during simulation (e.g., picked up by gripper).
    """

    def __init__(
        self,
        prim_path,
        offset: np.ndarray,
        name="rigid_prim",
        position=None,
        translation=None,
        orientation=None,
        scale=None,
        visible=None,
        reset_xform_properties=True,
        mass=None,
        density=None,
        linear_velocity=None,
        angular_velocity=None,
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
            mass,
            density,
            linear_velocity,
            angular_velocity,
        )
        self.offset = offset

    def get_world_pose(self):
        position, orientation = super().get_world_pose()
        return position + self.offset, orientation

    def get_world_pose_no_offset(self):
        return super().get_world_pose()

    def get_world_position_no_offset(self):
        return super().get_world_pose()[0][:2]
