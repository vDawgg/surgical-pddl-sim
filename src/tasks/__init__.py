from enum import StrEnum, auto
from typing import TYPE_CHECKING

from src.tasks.schemas import Problem

if TYPE_CHECKING:
    from src.base.task import DvrkBaseTask
    from src.plan.plan import Plan


class Task(StrEnum):
    NEEDLE_TRANSFER = auto()
    RING_AND_PEG = auto()


def get_task(task_name: str, problem: Problem) -> "DvrkBaseTask":
    from src.tasks.needle_transfer import NeedleTransfer
    from src.tasks.ring_and_peg import RingAndPeg

    match task_name:
        case Task.NEEDLE_TRANSFER:
            return NeedleTransfer(name="needle_transfer_task", problem=problem)
        case Task.RING_AND_PEG:
            return RingAndPeg(name="ring_and_peg_task", problem=problem)
