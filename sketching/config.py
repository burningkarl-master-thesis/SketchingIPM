import dacite
import dataclasses
import enum
import itertools
import json
import typing


T = typing.TypeVar("T")


class Preconditioning(enum.Enum):
    NONE = "none"
    QR = "qr"
    SPARSE_QR = "sparse_qr"
    CHOLESKY = "cholesky"


class DaciteFromFile:
    @classmethod
    def from_file(cls: typing.Type[T], filename: str) -> T:
        """ Generates a SketchingConfig from a JSON configuration file """
        with open(filename) as f:
            json_config = json.load(f)
        return dacite.from_dict(
            cls, json_config, dacite.Config(strict=True, cast=[enum.Enum])
        )


@dataclasses.dataclass(frozen=True)
class ProblemConfig(DaciteFromFile):
    """ Stores all configuration options for the problem instance """

    m: int = 100
    n: int = 10000
    nnz_factor: float = 2
    density: float = dataclasses.field(init=False)
    mu_min_exponent: float = -12
    mu_max_exponent: float = 0
    mu_steps: int = 10

    def __post_init__(self):
        """ Validates the current configuration """
        try:
            assert 1 <= self.m <= self.n
            assert 0 <= self.nnz_factor <= self.m
        except AssertionError as error:
            raise ValueError("Invalid value in configuration") from error

        # nnz(A) = self.nnz_factor * self.n
        object.__setattr__(self, "density", self.nnz_factor / self.m)


@dataclasses.dataclass(frozen=True)
class SketchingConfig(DaciteFromFile):
    """ Stores all configuration options for the sketching experiments """

    w_factor: float = 1.5
    s: int = 3

    def __post_init__(self):
        """ Validates the current configuration """
        try:
            assert self.w_factor >= 1
            assert self.s >= 1
        except AssertionError as error:
            raise ValueError("Invalid value in configuration") from error


@dataclasses.dataclass(frozen=True)
class PreconditioningConfig(DaciteFromFile):
    preconditioning: Preconditioning = Preconditioning.QR


def make_product_class(cls: typing.Type[T]):
    """ Creates a new dataclass turning all fields of type T into type List[T] """
    # The following construction assumes that all fields of cls have a default, not a default_factory
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
        for field in dataclasses.fields(cls)
        if field.init
    ]

    def configs(self) -> T:
        """ Generates a config object for each possible combination of the given parameter values """
        keys, values = zip(*dataclasses.asdict(self).items())
        for values_combination in itertools.product(*values):
            yield cls(**dict(zip(keys, values_combination)))

    ConfigProduct = dataclasses.make_dataclass(
        cls_name=cls.__name__ + "Product",
        fields=list_config_fields,
        bases=(DaciteFromFile,),
        namespace={"configs": configs},
        frozen=True,
    )

    return ConfigProduct


ProblemConfigProduct = make_product_class(ProblemConfig)
SketchingConfigProduct = make_product_class(SketchingConfig)
PreconditioningConfigProduct = make_product_class(PreconditioningConfig)


@dataclasses.dataclass(frozen=True)
class ExperimentConfig(DaciteFromFile):
    """ Stores all configuration options concerning all experiments """

    number_of_runs: int = 1
    group: str = ""
    problem_config_product: ProblemConfigProduct = ProblemConfigProduct()
    sketching_config_product: SketchingConfigProduct = SketchingConfigProduct()
    preconditioning_config_product: PreconditioningConfigProduct = (
        PreconditioningConfigProduct()
    )

    def __post_init__(self):
        """ Validates the current configuration """
        try:
            assert self.number_of_runs >= 1
        except AssertionError as error:
            raise ValueError("Invalid value in configuration") from error

    def problem_configs(self) -> typing.Iterator[ProblemConfig]:
        yield from self.problem_config_product.configs()

    def sketching_configs(self) -> typing.Iterator[SketchingConfig]:
        yield from self.sketching_config_product.configs()

    def preconditioning_configs(self) -> typing.Iterator[PreconditioningConfig]:
        yield from self.preconditioning_config_product.configs()
