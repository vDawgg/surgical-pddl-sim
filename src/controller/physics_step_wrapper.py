import numpy as np
from typing import Optional
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot

from src.controller.lula_controller import LulaController


class PhysicsStepWrapper:
    def __init__(self, controller: LulaController, robot: Robot, world: World):
        self._controller = controller
        self._robot = robot
        self._world = world

    def step(
        self,
        action_type: str,
        target_position: np.ndarray | None = None,
        target_orientation: np.ndarray | None = None,
    ):
        """
        Applies one step of the controller and advances the simulation.
        """
        action = self._controller.forward(
            action_type, target_position, target_orientation
        )
        self._robot.apply_action(action)
        self._world.step(render=True)
