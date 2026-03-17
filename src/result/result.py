from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class Result:
    image_start: str | None
    image_end: str | None
    success: bool


@dataclass
class Results:
    results: list[Result]

    def save_results(self, results: pl.DataFrame, results_file_path: Path):
        sim_results = pl.DataFrame(result.__dict__ for result in self.results)
        results.hstack(sim_results).write_csv(results_file_path)
