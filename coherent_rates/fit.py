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

    @abstractmethod
    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> T:
        ...

    @classmethod
    @abstractmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: T,
    ) -> tuple[float, ...]:
        ...

    def get_rates_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> tuple[float, ...]:
        fit = self.get_fit_from_isf(data)
        return self.get_rates_from_fit(fit)

    @classmethod
    def get_function_for_fit(
        cls: type[Self],
        fit: T,
    ) -> Callable[[_BT0], ValueList[_BT0]]:
        return functools.partial(cls.get_fitted_data, fit)

    @classmethod
    @abstractmethod
    def get_fitted_data(
        cls: type[Self],
        fit: T,
        basis: _BT0,
    ) -> ValueList[_BT0]:
        ...

    @staticmethod
    @abstractmethod
    def get_rate_labels() -> tuple[str, ...]:
        ...

    @classmethod
    def n_rates(cls: type[Self]) -> int:
        return len(cls.get_rate_labels())

    @abstractmethod
    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        ...


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

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> GaussianParameters:
        # truncate value list

        is_increasing = np.diff(np.abs(data["data"])) > 0
        first_increasing_idx = np.argmax(is_increasing).item()
        idx = data["basis"].n - 1 if first_increasing_idx == 0 else first_increasing_idx
        isf = _truncate_value_list(data, idx)

        # Gaussian fitting
        def gaussian(
            x: np.ndarray[Any, np.dtype[np.float64]],
            a: float,
            b: float,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return (1 - a) + a * np.exp(-1 * np.square(x / b) / 2)

        y_data = get_measured_data(isf["data"], measure="abs")
        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                gaussian,
                isf["basis"].times,
                y_data,
                bounds=([0, -np.inf], [1, np.inf]),
            ),
        )

        return GaussianParameters(parameters[0], np.abs(parameters[1]))

    @classmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: GaussianParameters,
    ) -> tuple[float,]:
        return (1 / fit.width,)

    @classmethod
    def get_fitted_data(
        cls: type[Self],
        fit: GaussianParameters,
        basis: _BT0,
    ) -> ValueList[_BT0]:
        gaussian = fit
        y_fit = (1 - gaussian.amplitude) + gaussian.amplitude * np.exp(
            -1 * np.square(basis.times / gaussian.width) / 2,
        )
        return {"basis": basis, "data": y_fit.astype(np.complex128)}

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
        return hash("DoubleGaussianMethod")

    @classmethod
    def get_fitted_data(
        cls: type[Self],
        fit: tuple[GaussianParameters, GaussianParameters],
        basis: _BT0,
    ) -> ValueList[_BT0]:
        gaussian1, gaussian2 = fit
        y_fit = (
            (1 - gaussian1.amplitude - gaussian2.amplitude)
            + gaussian1.amplitude
            * np.exp(-1 * np.square(basis.times / gaussian1.width) / 2)
            + gaussian2.amplitude
            * np.exp(-1 * np.square(basis.times / gaussian2.width) / 2)
        )
        return {"basis": basis, "data": y_fit.astype(np.complex128)}

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> tuple[GaussianParameters, GaussianParameters]:
        # Double gaussian fitting
        def double_gaussian(
            x: np.ndarray[Any, np.dtype[np.float64]],
            a: float,
            b: float,
            c: float,
            d: float,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return (
                (1 - a - c)
                + a * np.exp(-1 * np.square(x / b) / 2)
                + c * np.exp(-1 * np.square(x / d) / 2)
                - 1000 * max(a + c - 1, 0)
            )

        y_data = get_measured_data(data["data"], measure="abs")
        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                double_gaussian,
                data["basis"].times,
                y_data,
                bounds=([0, -np.inf, 0, -np.inf], [1, np.inf, 1, np.inf]),
            ),
        )

        return (
            GaussianParameters(parameters[0], np.abs(parameters[1])),
            GaussianParameters(parameters[2], np.abs(parameters[3])),
        )

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
        return hash("ExponentialMethod")

    @classmethod
    def get_fitted_data(
        cls: type[Self],
        fit: ExponentialParameters,
        basis: _BT0,
    ) -> ValueList[_BT0]:
        times = basis.times
        exponential = fit
        y_fit = (1 - exponential.amplitude) + exponential.amplitude * np.exp(
            -1 * times / exponential.time_constant,
        )
        return {"basis": basis, "data": y_fit.astype(np.complex128)}

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> ExponentialParameters:
        # Exponential fitting
        def exponential(
            x: np.ndarray[Any, np.dtype[np.float64]],
            a: float,
            b: float,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return (1 - a) + a * np.exp(-1 * x / b)

        y_data = get_measured_data(data["data"], measure="abs")
        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                exponential,
                data["basis"].times,
                y_data,
                bounds=([0, -np.inf], [1, np.inf]),
            ),
        )

        return ExponentialParameters(parameters[0], parameters[1])

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


class GaussianPlusExponentialMethod(
    FitMethod[tuple[GaussianParameters, ExponentialParameters]],
):
    """Fit the data to a gaussian plus an exponential."""

    def __hash__(self: Self) -> int:
        return hash("GaussianPlusExponentialMethod")

    @classmethod
    def get_fitted_data(
        cls: type[Self],
        fit: tuple[GaussianParameters, ExponentialParameters],
        basis: _BT0,
    ) -> ValueList[_BT0]:
        times = basis.times
        gaussian, exponential = fit
        y_fit = (
            (1 - gaussian.amplitude - exponential.amplitude)
            + gaussian.amplitude * np.exp(-1 * np.square(times / gaussian.width) / 2)
            + exponential.amplitude * np.exp(-1 * times / exponential.time_constant)
        )
        return {"basis": basis, "data": y_fit.astype(np.complex128)}

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> tuple[GaussianParameters, ExponentialParameters]:
        # Gaussian with exponential fitting
        def gaussian_and_exp(
            x: np.ndarray[Any, np.dtype[np.float64]],
            a: float,
            b: float,
            c: float,
            d: float,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return (
                (1 - a - c)
                + a * np.exp(-1 * np.square(x / b) / 2)
                + c * np.exp(-1 * x / d)
                - 1000 * max(a + c - 1, 0)
            )

        y_data = get_measured_data(data["data"], measure="abs")
        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                gaussian_and_exp,
                data["basis"].times,
                y_data,
                bounds=([0, -np.inf, 0, -np.inf], [1, np.inf, 1, np.inf]),
            ),
        )

        return (
            GaussianParameters(parameters[0], np.abs(parameters[1])),
            ExponentialParameters(parameters[2], parameters[3]),
        )

    @classmethod
    def get_rates_from_fit(
        cls: type[Self],
        fit: tuple[GaussianParameters, ExponentialParameters],
    ) -> tuple[float, float]:
        return (1 / fit[0].width, 1 / fit[1].time_constant)

    def get_fit_curve(
        self: Self,
        data: ValueList[_BT0],
    ) -> ValueList[_BT0]:
        times = data["basis"].times
        gaussian, exponential = self.get_fit_from_isf(data)
        y_fit = (
            (1 - gaussian.amplitude - exponential.amplitude)
            + gaussian.amplitude * np.exp(-1 * np.square(times / gaussian.width) / 2)
            + exponential.amplitude * np.exp(-1 * times / exponential.time_constant)
        )
        return {"basis": data["basis"], "data": y_fit.astype(np.complex128)}

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
