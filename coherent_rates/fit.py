from __future__ import annotations

import functools
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Self, TypeVar, cast

import numpy as np
from scipy.constants import Boltzmann  # type: ignore library type
from scipy.optimize import curve_fit  # type: ignore library type
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    ExplicitTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.util import get_measured_data

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList

    from coherent_rates.system import PeriodicSystem, PeriodicSystemConfig


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])
T = TypeVar("T")


class FitMethod(ABC, Generic[T]):
    """A method used for fitting an ISF."""

    @abstractmethod
    def __hash__(self: Self) -> int:
        ...

    @classmethod
    @abstractmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: T,
    ) -> tuple[float, ...]:
        ...

    @staticmethod
    @abstractmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        ...

    @staticmethod
    @abstractmethod
    def _params_from_fit(
        fit: T,
    ) -> tuple[float, ...]:
        ...

    @staticmethod
    @abstractmethod
    def _fit_from_params(
        dt: float,
        *params: *tuple[float, ...],
    ) -> T:
        ...

    @staticmethod
    @abstractmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        ...

    @classmethod
    @abstractmethod
    def _fit_param_initial_guess(cls: type[Self]) -> list[float]:
        ...

    @staticmethod
    @abstractmethod
    def get_rate_labels() -> tuple[str, ...]:
        ...

    @abstractmethod
    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        ...

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> T:
        y_data = get_measured_data(data["data"], measure="abs")
        dt = np.max(data["basis"].times) / data["basis"].times.size
        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                self._fit_fn,
                data["basis"].times / dt,
                y_data,
                p0=self._fit_param_initial_guess(),
                bounds=self._fit_param_bounds(),
            ),
        )

        return self._fit_from_params(dt.item(), *parameters)

    def get_rates_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> tuple[float, ...]:
        fit = self.get_fit_from_isf(data)
        return self.get_rates_from_fit(fit)

    @classmethod
    def get_fitted_data(
        cls: type[Self],
        fit: T,
        basis: _BT0,
    ) -> ValueList[_BT0]:
        data = cls._fit_fn(basis.times, *cls._params_from_fit(fit))
        return {"basis": basis, "data": data.astype(np.complex128)}

    @classmethod
    def get_function_for_fit(
        cls: type[Self],
        fit: T,
    ) -> Callable[[_BT0], ValueList[_BT0]]:
        return functools.partial(cls.get_fitted_data, fit)

    @classmethod
    def n_params(cls: type[Self]) -> int:
        return len(cls._fit_param_initial_guess())

    @classmethod
    def n_rates(cls: type[Self]) -> int:
        return len(cls.get_rate_labels())


def _truncate_value_list(
    values: ValueList[BasisWithTimeLike[int, int]],
    index: int,
) -> ValueList[ExplicitTimeBasis[int]]:
    data = values["data"][0 : index + 1]
    new_times = ExplicitTimeBasis(values["basis"].times[0 : index + 1])
    return {"basis": new_times, "data": data}


def _get_free_particle_time(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    direction: tuple[int, ...] | None = None,
) -> float:
    direction = tuple(1 for _ in config.shape) if direction is None else direction
    basis = system.get_potential(config.shape, config.resolution)["basis"]
    dk_stacked = BasisUtil(basis).dk_stacked

    k = np.linalg.norm(np.einsum("i,ij->j", direction, dk_stacked))  # type:ignore unknown lib type
    k = np.linalg.norm(dk_stacked[0]) if k == 0 else k

    return np.sqrt(system.mass / (Boltzmann * config.temperature * k**2))


@dataclass
class GaussianParameters:
    """parameters for a gaussian fit."""

    amplitude: float
    width: float


class GaussianMethod(FitMethod[GaussianParameters]):
    """Fit the data to a single Gaussian."""

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(b"GaussianMethod")
        return int.from_bytes(h.digest(), "big")

    @staticmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        a, b = params
        return (1 - a) + a * np.exp(-1 * np.square(x / b) / 2)

    @staticmethod
    def _params_from_fit(
        fit: GaussianParameters,
    ) -> tuple[float, float]:
        return (fit.amplitude, fit.width)

    @staticmethod
    def _fit_from_params(
        dt: float,
        *params: *tuple[float, ...],
    ) -> GaussianParameters:
        return GaussianParameters(params[0], dt * np.abs(params[1]))

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, -np.inf], [1, np.inf])

    @classmethod
    def _fit_param_initial_guess(cls: type[Self]) -> list[float]:
        return [1, 1]

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> GaussianParameters:
        # Stop trying to fit past the first non-decreasing ISF
        is_increasing = np.diff(np.abs(data["data"])) > 0
        first_increasing_idx = np.argmax(is_increasing).item()
        idx = data["basis"].n - 1 if first_increasing_idx == 0 else first_increasing_idx
        truncated = _truncate_value_list(data, idx)

        return super().get_fit_from_isf(truncated)

    @classmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: GaussianParameters,
    ) -> tuple[float,]:
        return (1 / fit.width,)

    @staticmethod
    def get_rate_labels() -> tuple[str]:
        return ("Gaussian",)

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 4 * _get_free_particle_time(system, config, direction)


class DoubleGaussianMethod(FitMethod[tuple[GaussianParameters, GaussianParameters]]):
    """Fit the data to a double Gaussian."""

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(b"DoubleGaussianMethod")
        return int.from_bytes(h.digest(), "big")

    @staticmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        a, b, c, d = params
        return (
            (1 - a - c)
            + a * np.exp(-1 * np.square(x / b) / 2)
            + c * np.exp(-1 * np.square(x / d) / 2)
            - 1000 * max(a + c - 1, 0)
        )

    @classmethod
    def _fit_param_initial_guess(cls: type[Self]) -> list[float]:
        return [0.5, 1, 0.5, 2]

    @staticmethod
    def _params_from_fit(
        fit: tuple[GaussianParameters, GaussianParameters],
    ) -> tuple[float, float, float, float]:
        return (fit[0].amplitude, fit[0].width, fit[1].amplitude, fit[1].width)

    @staticmethod
    def _fit_from_params(
        dt: float,
        *params: *tuple[float, ...],
    ) -> tuple[GaussianParameters, GaussianParameters]:
        return (
            GaussianParameters(params[0], dt * np.abs(params[1])),
            GaussianParameters(params[2], dt * np.abs(params[3])),
        )

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, -np.inf, 0, -np.inf], [1, np.inf, 1, np.inf])

    @classmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: tuple[GaussianParameters, GaussianParameters],
    ) -> tuple[float, float]:
        return (
            max(1 / fit[0].width, 1 / fit[1].width),
            min(1 / fit[0].width, 1 / fit[1].width),
        )

    @staticmethod
    def get_rate_labels() -> tuple[str, str]:
        return ("Fast gaussian", "Slow gaussian")

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 40 * _get_free_particle_time(system, config, direction)


@dataclass
class ExponentialParameters:
    """Parameters of an exponential fit."""

    amplitude: float
    time_constant: float


class ExponentialMethod(FitMethod[ExponentialParameters]):
    """Fit the data to an exponential."""

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(b"ExponentialMethod")
        return int.from_bytes(h.digest(), "big")

    @staticmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        a, b = params
        return (1 - a) + a * np.exp(-1 * x / b)

    @staticmethod
    def _params_from_fit(
        fit: ExponentialParameters,
    ) -> tuple[float, float]:
        return (fit.amplitude, fit.time_constant)

    @staticmethod
    def _fit_from_params(
        dt: float,
        *params: *tuple[float, ...],
    ) -> ExponentialParameters:
        return ExponentialParameters(params[0], dt * params[1])

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, -np.inf], [1, np.inf])

    @classmethod
    def _fit_param_initial_guess(cls: type[Self]) -> list[float]:
        return [1, 1]

    @classmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: ExponentialParameters,
    ) -> tuple[float,]:
        return (1 / fit.time_constant,)

    @staticmethod
    def get_rate_labels() -> tuple[str]:
        return ("Exponential",)

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 40 * _get_free_particle_time(system, config, direction)

    @classmethod
    def n_params(cls: type[Self]) -> int:
        return 2


class GaussianPlusExponentialMethod(
    FitMethod[tuple[GaussianParameters, ExponentialParameters]],
):
    """Fit the data to a gaussian plus an exponential."""

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(b"GaussianPlusExponentialMethod")
        return int.from_bytes(h.digest(), "big")

    @staticmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        a, b, c, d = params
        return (
            (1 - a - c)
            + a * np.exp(-1 * np.square(x / b) / 2)
            + c * np.exp(-1 * x / d)
            - 1000 * max(np.sign(b - d), 0)
            - 1000 * max(a + c - 1, 0)
        )

    @staticmethod
    def _params_from_fit(
        fit: tuple[GaussianParameters, ExponentialParameters],
    ) -> tuple[float, float, float, float]:
        return (fit[0].amplitude, fit[0].width, fit[1].amplitude, fit[1].time_constant)

    @staticmethod
    def _fit_from_params(
        dt: float,
        *params: *tuple[float, ...],
    ) -> tuple[GaussianParameters, ExponentialParameters]:
        return (
            GaussianParameters(params[0], dt * np.abs(params[1])),
            ExponentialParameters(params[2], dt * params[3]),
        )

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, -np.inf, 0, -np.inf], [1, np.inf, 1, np.inf])

    @classmethod
    def _fit_param_initial_guess(cls: type[Self]) -> list[float]:
        return [0.5, 1, 0.5, 1]

    @classmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: tuple[GaussianParameters, ExponentialParameters],
    ) -> tuple[float, float]:
        return (1 / fit[0].width, 1 / fit[1].time_constant)

    @staticmethod
    def get_rate_labels() -> tuple[str, str]:
        return ("Gaussian", "Exponential")

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 40 * _get_free_particle_time(system, config, direction)
