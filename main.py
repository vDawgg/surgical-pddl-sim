from isaacsim.simulation_app import SimulationApp

simulation_app: SimulationApp = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import RigidPrim
from isaacsim.util.debug_draw import _debug_draw

from src.tasks.needle_transfer import NeedleTransfer
from src.controller.lula_controller import LulaController
from src.controller.physics_step_wrapper import PhysicsStepWrapper
from src.controller.lula_planner import LulaMotionPlanner


if __name__ == "__main__":
    ## Simulation setup
    world: World = World()
    task = NeedleTransfer(name="needle_transfer_task")
    world.add_task(task)

    world.reset()

    dvrk: Robot = world.scene.get_object("dvrk")
    planner = LulaMotionPlanner(robot=dvrk)
    controller = LulaController(
        name="psm_controller",
        robot_articulation=dvrk,
        gripper_positions=task.gripper_positions,
        planner=planner,
    )
    wrapper = PhysicsStepWrapper(controller=controller, robot=dvrk, world=world)

    obs = task.get_observations()

    needle_pose = obs["needle"]
    action_sequence = [
        # TODO: We still need to find a cleaner way than adding an additional 'move' action for the robot to cleanly
        #       pick up the item
        # TODO: Ideally we can specify the name of the position here, if it is not an open/close action and let
        #       the task fill in the position information for this object. This will only be possible if the above
        #       is completed
        ("move", (needle_pose[0] + np.array([0.0, 0.0, 0.05]), needle_pose[1])),
        ("move", needle_pose),
        ("pick", (None, None)),
        ("move", (np.array([0.1, 0.1, 0.1]), None)),
        ("place", (None, None)),
    ]
    world.reset()

    psm_tool_tip = RigidPrim("/World/dVRK/psm_tool_tip_link", "psm_tool_tip_view")
    psm_tool_tip_pos = psm_tool_tip.get_world_poses()[0][0]

    # TODO: This should be a command-line argument
    num_trials = 2
    for trial in range(num_trials):
        print(f"{'=' * 60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'=' * 60}")

        for action_type, target in action_sequence:
            print(f"Executing action: {action_type} to {target}")

            max_steps = 200
            tolerance = 0.002
            # TODO: We need to find general stop-criteria that we can communicate via the task and/or for each step
            #       to ensure a cleaner execution-flow
            for step in range(max_steps):
                wrapper.step(action_type, *target)
                if action_type == "move":
                    # TODO: Wrap this in the physics wrapper and implement something similar for the pick and place
                    #       actions
                    #       -> Additionally it would be a lot cleaner to just check whether rmpflow thinks it has arrived
                    #          at the position.
                    distance = np.linalg.norm(target[0] - psm_tool_tip_pos)
                    if distance <= tolerance:
                        break
        world.reset()

    print(f"\\n{'=' * 60}")
    print("Simulation completed successfully!")
    print(f"{'=' * 60}")
    simulation_app.close()
