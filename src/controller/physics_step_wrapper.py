import numpy as np
from typing import Optional
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot

from src.controller.lula_controller import LulaController
from src.plan.plan import Action


class PhysicsStepWrapper:
    def __init__(self, controller: LulaController, robot: Robot, world: World):
        self._controller = controller
        self._robot = robot
        self._world = world

    # TODO: Think about dropping this and moving all of this to either main or the controller
    def step(self, action: Action):
        """
        Applies one step of the controller and advances the simulation.
        """
        action = self._controller.forward(action)
        self._robot.apply_action(action)
        self._world.step(render=True)
