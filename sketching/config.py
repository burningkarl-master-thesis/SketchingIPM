import dacite
import dataclasses
import enum
import itertools
import json
import typing


class Preconditioning(enum.Enum):
    NONE = "none"
    QR = "qr"
    SPARSE_QR = "sparse_qr"
    CHOLESKY = "cholesky"


@dataclasses.dataclass(frozen=True)
class SketchingConfig:
    """ Stores all configuration options for the sketching experiments """

    number_of_runs: int = 1
    m: int = 100
    n: int = 10000
    nnz_factor: float = 2
    density: float = dataclasses.field(init=False)
    preconditioning: Preconditioning = Preconditioning.QR
    w_factor: float = 1.5
    w: int = dataclasses.field(init=False)
    s: int = 3

    @classmethod
    def from_file(cls, filename: str) -> "SketchingConfig":
        """ Generates a SketchingConfig from a JSON configuration file """
        with open(filename) as f:
            json_config = json.load(f)
        return dacite.from_dict(
            cls, json_config, dacite.Config(strict=True, cast=[enum.Enum])
        )

    def __post_init__(self):
        """ Validates the current configuration """
        try:
            assert self.number_of_runs >= 1
            assert 1 <= self.m <= self.n
            assert 0 <= self.nnz_factor <= self.m
            assert self.w_factor >= 1
            assert self.s >= 1
        except AssertionError as error:
            raise ValueError("Invalid value in configuration") from error

        # nnz(A) = self.nnz_factor * self.n
        object.__setattr__(self, "density", self.nnz_factor / self.m)

        # Set w according to the given w_factor
        object.__setattr__(self, "w", int(self.w_factor * self.m))


# The following construction assumes that all fields of SketchingConfig have a default, not a default_factory
list_config_fields = [
    (
        field.name,
        typing.List[field.type],
        dataclasses.field(
            default_factory=lambda default=field.default: [default],
            init=True,
            repr=field.repr,
            hash=field.hash,
            compare=field.compare,
            metadata=field.metadata,
        ),
    )
    for field in dataclasses.fields(SketchingConfig)
    if field.init
]
SketchingConfigProductSuper = dataclasses.make_dataclass(
    "SketchingConfigProductSuper", list_config_fields, frozen=True
)


class SketchingConfigProduct(SketchingConfigProductSuper):
    """ Stores a list of possible parameter values for each option in SketchingConfig """

    @classmethod
    def from_file(cls, filename: str) -> "SketchingConfigProduct":
        """ Generates a SketchingConfigProduct from a JSON configuration file """
        with open(filename) as f:
            json_config = json.load(f)
        return dacite.from_dict(
            cls, json_config, dacite.Config(strict=True, cast=[enum.Enum])
        )

    def configs(self):
        """ Generates a SketchingConfig for each possible combination of the given parameter values """
        keys, values = zip(*dataclasses.asdict(self).items())
        for values_combination in itertools.product(*values):
            yield SketchingConfig(**dict(zip(keys, values_combination)))
