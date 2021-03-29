import dacite
import dataclasses
import enum
import json
from typing import Optional


class LinearSolver(enum.Enum):
    CG_METHOD = 'cg_method'
    CHOLESKY = 'cholesky'


@dataclasses.dataclass
class SketchingConfig:
    m: Optional[int] = 100
    n: Optional[int] = 10000
    nnz_factor: Optional[float] = 2
    density: float = dataclasses.field(init=False)
    w_factor: Optional[float] = 1.5
    w: int = dataclasses.field(init=False)
    s: Optional[int] = 3
    linear_solver: Optional[LinearSolver] = LinearSolver.CG_METHOD
    cg_iterations: Optional[int] = 100

    @staticmethod
    def from_file(filename: str) -> 'SketchingConfig':
        """ Generates a SketchingConfig from a JSON configuration file """
        with open(filename) as f:
            json_config = json.load(f)
        return dacite.from_dict(SketchingConfig, json_config, dacite.Config(strict=True, cast=[enum.Enum]))

    def __post_init__(self):
        """ Validates the current configuration """
        try:
            assert 1 <= self.m <= self.n
            assert 0 <= self.nnz_factor <= self.m
            assert self.w_factor >= 1
            assert self.s >= 1
            assert self.cg_iterations >= 1
        except AssertionError as error:
            raise ValueError("Invalid value in configuration") from error

        # nnz(A) = self.nnz_factor * self.n
        self.density = self.nnz_factor / self.m

        # Set w according to the given w_factor
        self.w = int(self.w_factor * self.m)
