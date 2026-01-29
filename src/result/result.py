from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class Result:
    plan_file: str
    image_start: str
    image_end: str | None
    success: bool


@dataclass
class Results:
    results: list[Result]

    def save_results(self, plan_dir: Path):
        pl.DataFrame(result.__dict__ for result in self.results).write_csv(
            plan_dir / "results.csv"
        )
