from enum import StrEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.base.task import DvrkTask
    from src.plan.plan import Plan


class Task(StrEnum):
    NEEDLE_TRANSFER = auto()
    RING_AND_PEG = auto()


def get_task(task_name: str) -> "DvrkTask":
    from src.tasks.needle_transfer import NeedleTransfer
    from src.tasks.peg_and_ring import PegAndRing

    match task_name:
        case Task.NEEDLE_TRANSFER:
            return NeedleTransfer(name="needle_transfer_task")
        case Task.RING_AND_PEG:
            return PegAndRing(name="peg_and_ring_task")


def get_plan(task: "DvrkTask") -> "Plan":
    from src.plan.plan import Action, ActionType, Plan
    from src.tasks.needle_transfer import NeedleTransfer
    from src.tasks.peg_and_ring import PegAndRing

    if type(task) is NeedleTransfer:
        return Plan.from_actions(
            [
                Action.from_prim(ActionType.MOVE, task.needle),
                Action.from_prim(ActionType.PICK),
                Action.from_prim(ActionType.MOVE, task.goal),
                Action.from_prim(ActionType.PLACE),
            ]
        )
    elif type(task) is PegAndRing:
        return Plan.from_actions(
            [
                Action.from_prim(ActionType.MOVE, task.blue_ring),
                Action.from_prim(ActionType.PICK),
                Action.from_prim(ActionType.MOVE, task.blue_peg),
                Action.from_prim(ActionType.PLACE),
                Action.from_prim(ActionType.MOVE, task.green_ring),
                Action.from_prim(ActionType.PICK),
                Action.from_prim(ActionType.MOVE, task.green_peg),
                Action.from_prim(ActionType.PLACE),
                Action.from_prim(ActionType.MOVE, task.red_ring),
                Action.from_prim(ActionType.PICK),
                Action.from_prim(ActionType.MOVE, task.red_peg),
                Action.from_prim(ActionType.PLACE),
            ]
        )
