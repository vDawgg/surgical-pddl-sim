import numpy as np
from isaacsim.core.prims import SingleXFormPrim


class OffsetPrim(SingleXFormPrim):
    """
    Wrapper around prims to allow for the definition of an offset to the world position.
    This ensures proper placement of the at grippable positions of the object.
    """

    def __init__(
        self,
        prim_path,
        offset: np.ndarray,
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
        self.offset = offset

    def get_world_pose(self):
        position, orientation = super().get_world_pose()
        return position + self.offset, orientation
