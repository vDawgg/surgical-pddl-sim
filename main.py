from isaacsim.simulation_app import SimulationApp

simulation_app: SimulationApp = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World

from standalone.tasks.standard import StandardTask


if __name__ == "__main__":
    ## Simulation setup
    world: World = World()
    world.add_task(StandardTask(name="standard_task"))

    world.reset()

    ## Simulation start
    while True:
        world.step(render=True)
