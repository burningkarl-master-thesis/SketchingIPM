import dacite
import dataclasses
import enum
import json
from typing import Optional


class Preconditioning(enum.Enum):
    NONE = 'none'
    SKETCHING = 'sketching'
    FULL = 'full'


class LinearSolver(enum.Enum):
    CG_METHOD = 'cg_method'
    CHOLESKY = 'cholesky'


@dataclasses.dataclass(frozen=True)
class SketchingConfig:
    number_of_runs: int = 10
    m: Optional[int] = 100
    n: Optional[int] = 10000
    nnz_factor: Optional[float] = 2
    density: float = dataclasses.field(init=False)
    preconditioning: Preconditioning = Preconditioning.SKETCHING
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
            assert self.number_of_runs >= 1
            assert 1 <= self.m <= self.n
            assert 0 <= self.nnz_factor <= self.m
            assert self.w_factor >= 1
            assert self.s >= 1
            assert self.cg_iterations >= 1
        except AssertionError as error:
            raise ValueError("Invalid value in configuration") from error

        # nnz(A) = self.nnz_factor * self.n
        object.__setattr__(self, 'density', self.nnz_factor / self.m)

        # Set w according to the given w_factor
        object.__setattr__(self, 'w', int(self.w_factor * self.m))
