from argparse import ArgumentParser
from src.tasks import Task

# NOTE: We need to execute the parsing step here as this will otherwise be ignored due to the sim-app loading
parser = ArgumentParser()
parser.add_argument("--task", choices=[t.value for t in Task], required=True)
parser.add_argument("--num_trials", default=2)
args = parser.parse_args()

from isaacsim.simulation_app import SimulationApp

simulation_app: SimulationApp = SimulationApp({"headless": False})

# NOTE: Suppress scientific notation for debug prints of np arrays
import numpy as np

np.set_printoptions(suppress=True)

from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim

from src.controller.lula_controller import LulaController
from src.controller.physics_step_wrapper import PhysicsStepWrapper
from src.controller.lula_planner import LulaMotionPlanner
from src.tasks import get_plan, get_task


if __name__ == "__main__":
    task_name = args.task
    num_trials = int(args.num_trials)
    task = get_task(task_name)

    ## Simulation setup
    world: World = World()
    world.add_task(task)
    world.reset()

    dvrk = task._dvrk
    planner = LulaMotionPlanner(robot=dvrk)
    controller = LulaController(
        name="psm_controller",
        robot_articulation=dvrk,
        gripper_positions=task.gripper_positions,
        planner=planner,
    )
    wrapper = PhysicsStepWrapper(controller=controller, robot=dvrk, world=world)

    world.reset()

    psm_tool_tip = RigidPrim("/World/dVRK/psm_tool_tip_link", "psm_tool_tip_view")
    psm_tool_tip_pos = psm_tool_tip.get_world_poses()[0][0]

    plan = get_plan(task)
    plan.add_via_points_to_plan()
    print(
        [
            f"{action.action_type}, {action.target_position}"
            for action in plan.action_sequence
        ]
    )
    for trial in range(num_trials):
        print(f"{'=' * 60}")
        print(f"TRIAL {trial + 1}/{num_trials}")
        print(f"{'=' * 60}")

        for action in plan.action_sequence:
            print(f"Executing action: {action.action_type}")
            if action.target_position is not None:
                print("Target position:", action.target_position)
                print("Target orientation:", action.target_orientation)

            max_steps = 300
            # TODO: We need to find general stop-criteria that we can communicate via the task and/or for each step
            #       to ensure a cleaner execution-flow
            #       -> Ideally this will be added to the controller
            for step in range(max_steps):
                wrapper.step(action)
            print("Current EE position:", planner.get_end_effector_pose()[0])
            print(f"{'-' * 60}")
        world.reset()

    print(f"{'=' * 60}")
    print("Simulation completed successfully!")
    print(f"{'=' * 60}")
    simulation_app.close()
