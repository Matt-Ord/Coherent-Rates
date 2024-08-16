from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Self, TypeVar

import numpy as np
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
from surface_potential_analysis.basis.time_basis_like import (
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.util.util import get_measured_data

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList

    from coherent_rates.system import PeriodicSystem, PeriodicSystemConfig


_BT0 = TypeVar("_BT0", bound=EvenlySpacedTimeBasis[Any, Any, Any])
T = TypeVar("T")


class FitMethod(Generic[T]):
    def fit_isf_data(
        self: Self,
        data: ValueList[_BT0],
    ) -> T:
        ...

    def get_rates_from_fit(
        self: Self,
        fit: T,
    ) -> tuple[float, ...]:
        ...

    def get_rates_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> tuple[float, ...]:
        fit = self.fit_isf_data(data)
        return self.get_rates_from_fit(fit)

    def get_fit_curve(
        self: Self,
        data: ValueList[_BT0],
    ) -> ValueList[_BT0]:
        ...

    def get_curve_label(
        self: Self,
    ) -> tuple[str, ...]:
        ...

    def plot_isf_with_fit(
        self: Self,
        data: ValueList[_BT0],
    ) -> None:
        y_fit = self.get_fit_curve(data)

        fig, ax, line = plot_value_list_against_time(data)
        line.set_label("ISF")

        fig, ax, line = plot_value_list_against_time(y_fit, ax=ax)
        line.set_label("Fit")

        ax.legend()
        fig.show()
        input()

    def n_rates(self: Self) -> int:
        ...

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        ...


def _truncate_value_list(
    values: ValueList[EvenlySpacedTimeBasis[int, int, int]],
    index: int,
) -> ValueList[EvenlySpacedTimeBasis[int, int, int]]:
    data = values["data"][0 : index + 1]
    new_times = EvenlySpacedTimeBasis(index + 1, 1, 0, values["basis"].times[index])
    return {"basis": new_times, "data": data}


def _get_free_particle_time(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    direction: tuple[int, ...] | None = None,
) -> float:
    direction = tuple(1 for _ in config.shape) if direction is None else direction
    basis = system.get_potential(config.shape, config.resolution)["basis"]
    dk_stacked = BasisUtil(basis).dk_stacked

    k = np.linalg.norm(np.einsum("i,ij->j", direction, dk_stacked))
    k = np.linalg.norm(dk_stacked[0]) if k == 0 else k

    return np.sqrt(system.mass / (Boltzmann * config.temperature * k**2))


@dataclass
class GaussianParameters:
    amplitude: float
    width: float


class GaussianMethod(FitMethod[GaussianParameters]):
    def fit_isf_data(
        self: Self,
        data: ValueList[_BT0],
    ) -> GaussianParameters:
        # truncate value list
        times = data["basis"]
        is_increasing = np.diff(np.abs(data["data"])) > 0
        first_increasing_idx = np.argmax(is_increasing).item()
        idx = times.n - 1 if first_increasing_idx == 0 else first_increasing_idx
        isf = _truncate_value_list(data, idx)

        times = EvenlySpacedTimeBasis(
            times.n,
            times.step,
            times.offset,
            times.times[idx],
        )

        # Gaussian fitting
        def gaussian(
            x: np.ndarray[Any, np.dtype[np.float64]],
            a: float,
            b: float,
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            return (1 - a) + a * np.exp(-1 * np.square(x / b) / 2)

        x_data = BasisUtil(isf["basis"]).nx_points
        y_data = get_measured_data(isf["data"], measure="abs")
        parameters, _covariance = curve_fit(
            gaussian,
            x_data,
            y_data,
            bounds=([0, -np.inf], [1, np.inf]),
        )
        fit_A = parameters[0]
        fit_B = parameters[1]
        dt = isf["basis"].times[1]
        return GaussianParameters(fit_A, np.abs(fit_B) * dt)

    def get_rates_from_fit(
        self: Self,
        fit: GaussianParameters,
    ) -> tuple[float,]:
        return (1 / fit.width,)

    def get_fit_curve(
        self: Self,
        data: ValueList[_BT0],
    ) -> ValueList[_BT0]:
        times = data["basis"].times
        gaussian = self.fit_isf_data(data)
        y_fit = (1 - gaussian.amplitude) + gaussian.amplitude * np.exp(
            -1 * np.square(times / gaussian.width) / 2,
        )
        return {"basis": data["basis"], "data": y_fit}

    def get_curve_label(
        self: Self,
    ) -> tuple[str,]:
        return ("Gaussian",)

    def n_rates(self: Self) -> int:
        return 1

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 4 * _get_free_particle_time(system, config, direction)


class DoubleGaussianMethod(FitMethod[tuple[GaussianParameters, GaussianParameters]]):
    def fit_isf_data(
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

        x_data = BasisUtil(data["basis"]).nx_points
        y_data = get_measured_data(data["data"], measure="abs")
        parameters, _covariance = curve_fit(
            double_gaussian,
            x_data,
            y_data,
            bounds=([0, -np.inf, 0, -np.inf], [1, np.inf, 1, np.inf]),
        )
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_C = parameters[2]
        fit_D = parameters[3]
        dt = data["basis"].times[1]
        return (
            GaussianParameters(fit_A, np.abs(fit_B) * dt),
            GaussianParameters(fit_C, np.abs(fit_D) * dt),
        )

    def get_rates_from_fit(
        self: Self,
        fit: tuple[GaussianParameters, GaussianParameters],
    ) -> tuple[float, float]:
        return (
            max(1 / fit[0].width, 1 / fit[1].width),
            min(1 / fit[0].width, 1 / fit[1].width),
        )

    def get_rates_from_isf(
        self: Self,
        data: ValueList[_BT0],
    ) -> tuple[float, float]:
        fit = self.fit_isf_data(data)
        return self.get_rates_from_fit(fit)

    def get_fit_curve(
        self: Self,
        data: ValueList[_BT0],
    ) -> ValueList[_BT0]:
        times = data["basis"].times
        gaussian1, gaussian2 = self.fit_isf_data(data)
        y_fit = (
            (1 - gaussian1.amplitude - gaussian2.amplitude)
            + gaussian1.amplitude * np.exp(-1 * np.square(times / gaussian1.width) / 2)
            + gaussian2.amplitude * np.exp(-1 * np.square(times / gaussian2.width) / 2)
        )
        return {"basis": data["basis"], "data": y_fit}

    def get_curve_label(
        self: Self,
    ) -> tuple[str, str]:
        return ("Fast gaussian", "Slow gaussian")

    def n_rates(self: Self) -> int:
        return 2

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 40 * _get_free_particle_time(system, config, direction)


@dataclass
class ExponentialParameters:
    amplitude: float
    time_constant: float


class ExponentialMethod(FitMethod[ExponentialParameters]):
    def fit_isf_data(
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

        x_data = BasisUtil(data["basis"]).nx_points
        y_data = get_measured_data(data["data"], measure="abs")
        parameters, _covariance = curve_fit(
            exponential,
            x_data,
            y_data,
            bounds=([0, -np.inf], [1, np.inf]),
        )
        fit_A = parameters[0]
        fit_B = parameters[1]
        dt = data["basis"].times[1]
        return ExponentialParameters(fit_A, fit_B * dt)

    def get_rates_from_fit(
        self: Self,
        fit: ExponentialParameters,
    ) -> tuple[float,]:
        return (1 / fit.time_constant,)

    def get_fit_curve(
        self: Self,
        data: ValueList[_BT0],
    ) -> ValueList[_BT0]:
        times = data["basis"].times
        exponential = self.fit_isf_data(data)
        y_fit = (1 - exponential.amplitude) + exponential.amplitude * np.exp(
            -1 * times / exponential.time_constant,
        )
        return {"basis": data["basis"], "data": y_fit}

    def get_curve_label(
        self: Self,
    ) -> tuple[str,]:
        return ("Exponential",)

    def n_rates(self: Self) -> int:
        return 1

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
    def fit_isf_data(
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

        x_data = BasisUtil(data["basis"]).nx_points
        y_data = get_measured_data(data["data"], measure="abs")
        parameters, _covariance = curve_fit(
            gaussian_and_exp,
            x_data,
            y_data,
            bounds=([0, -np.inf, 0, -np.inf], [1, np.inf, 1, np.inf]),
        )
        fit_A = parameters[0]
        fit_B = parameters[1]
        fit_C = parameters[2]
        fit_D = parameters[3]
        dt = data["basis"].times[1]
        return (
            GaussianParameters(fit_A, np.abs(fit_B) * dt),
            ExponentialParameters(fit_C, fit_D * dt),
        )

    def get_rates_from_fit(
        self: Self,
        fit: tuple[GaussianParameters, ExponentialParameters],
    ) -> tuple[float, float]:
        return (1 / fit[0].width, 1 / fit[1].time_constant)

    def get_fit_curve(
        self: Self,
        data: ValueList[_BT0],
    ) -> ValueList[_BT0]:
        times = data["basis"].times
        gaussian, exponential = self.fit_isf_data(data)
        y_fit = (
            (1 - gaussian.amplitude - exponential.amplitude)
            + gaussian.amplitude * np.exp(-1 * np.square(times / gaussian.width) / 2)
            + exponential.amplitude * np.exp(-1 * times / exponential.time_constant)
        )
        return {"basis": data["basis"], "data": y_fit}

    def get_curve_label(
        self: Self,
    ) -> tuple[str, str]:
        return ("Gaussian", "Exponential")

    def n_rates(self: Self) -> int:
        return 2

    def get_fit_time(
        self: Self,
        system: PeriodicSystem,
        config: PeriodicSystemConfig,
        direction: tuple[int, ...] | None = None,
    ) -> float:
        return 40 * _get_free_particle_time(system, config, direction)
