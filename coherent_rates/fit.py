from __future__ import annotations

import functools
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Self,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
)

import numpy as np
from scipy.constants import Boltzmann  # type: ignore library type
from scipy.optimize import curve_fit  # type: ignore library type
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
    ExplicitTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.util.util import get_measured_data

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList

    from coherent_rates.config import PeriodicSystemConfig
    from coherent_rates.system import System


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])
T = TypeVar("T")


class FitInfo(TypedDict):
    """Information about an ISF calculation."""

    system: System
    config: PeriodicSystemConfig


class FitMethod(ABC, Generic[T]):
    """A method used for fitting an ISF."""

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.get_rate_label().encode())
        return int.from_bytes(h.digest(), "big")

    @abstractmethod
    def get_rate_from_fit(
        self: Self,
        fit: T,
    ) -> float:
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
        *params: *tuple[float, ...],
    ) -> T:
        ...

    @staticmethod
    @abstractmethod
    def _scale_params(
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, ...]:
        ...

    @staticmethod
    @abstractmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        ...

    @classmethod
    @abstractmethod
    def _fit_param_initial_guess(
        cls: type[Self],
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> tuple[float, ...]:
        ...

    @abstractmethod
    def get_rate_label(self: Self) -> str:
        ...

    @abstractmethod
    def get_fit_times(
        self: Self,
        **info: Unpack[FitInfo],
    ) -> BasisWithTimeLike[Any, Any]:
        ...

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> T:
        y_data = get_measured_data(data["data"], measure="abs")
        delta_t = np.max(data["basis"].times) - np.min(data["basis"].times)
        dt = (delta_t / data["basis"].times.size).item()

        parameters, _covariance = cast(
            tuple[list[float], Any],
            curve_fit(
                self._fit_fn,
                data["basis"].times / dt,
                y_data,
                p0=self._scale_params(
                    1 / dt,
                    self._fit_param_initial_guess(data, **info),
                ),
                bounds=self._fit_param_bounds(),
            ),
        )

        return self._fit_from_params(*self._scale_params(dt, tuple(parameters)))

    def get_rate_from_isf(
        self: Self,
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> float:
        fit = self.get_fit_from_isf(data, **info)
        return self.get_rate_from_fit(fit)

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
        return len(cls._fit_param_bounds()[0])


def _truncate_value_list(
    values: ValueList[BasisWithTimeLike[int, int]],
    index: int,
) -> ValueList[ExplicitTimeBasis[int]]:
    data = values["data"][0 : index + 1]
    new_times = ExplicitTimeBasis(values["basis"].times[0 : index + 1])
    return {"basis": new_times, "data": data}


def get_free_particle_time(
    system: System,
    config: PeriodicSystemConfig,
) -> float:
    basis = system.get_potential(config.shape, config.resolution)["basis"]
    dk_stacked = BasisUtil(basis).dk_stacked

    k = np.linalg.norm(np.einsum("i,ij->j", config.direction, dk_stacked))  # type:ignore unknown lib type
    k = np.linalg.norm(dk_stacked[0]) if k == 0 else k

    return np.sqrt(system.mass / (Boltzmann * config.temperature * k**2))


@dataclass
class GaussianParameters:
    """parameters for a gaussian fit."""

    amplitude: float
    width: float


class GaussianMethod(FitMethod[GaussianParameters]):
    """Fit the data to a single Gaussian."""

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
        *params: *tuple[float, ...],
    ) -> GaussianParameters:
        return GaussianParameters(params[0], np.abs(params[1]))

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, 0], [1, np.inf])

    @classmethod
    def _fit_param_initial_guess(
        cls: type[Self],
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> tuple[float, float]:
        offset = 0.8 * np.min(np.abs(data["data"]))
        return (1 - offset, get_free_particle_time(**info))

    @staticmethod
    def _scale_params(
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, float]:
        return (params[0], dt * params[1])

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> GaussianParameters:
        # Stop trying to fit past the first non-decreasing ISF
        is_increasing = np.diff(np.abs(data["data"])) > 0
        first_increasing_idx = np.argmax(is_increasing).item()
        idx = data["basis"].n - 1 if first_increasing_idx == 0 else first_increasing_idx
        idx = max(idx, 10)
        truncated = _truncate_value_list(data, idx)

        return super().get_fit_from_isf(truncated, **info)

    def get_rate_from_fit(
        self: Self,
        fit: GaussianParameters,
    ) -> float:
        return 1 / fit.width

    def get_rate_label(self: Self) -> str:
        return "Gaussian"

    def get_fit_times(
        self: Self,
        **info: Unpack[FitInfo],
    ) -> EvenlySpacedTimeBasis[Any, Any, Any]:
        return EvenlySpacedTimeBasis(100, 1, 0, 4 * get_free_particle_time(**info))


@dataclass
class GaussianParametersWithOffset(GaussianParameters):
    """parameters for a gaussian fit."""

    offset: float


class GaussianMethodWithOffset(FitMethod[GaussianParametersWithOffset]):
    """Fit the data to a single Gaussian."""

    @staticmethod
    def _fit_fn(
        x: np.ndarray[Any, np.dtype[np.float64]],
        *params: *tuple[float, ...],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        a, b, offset = params
        return (
            offset
            + a * np.exp(-1 * np.square(x / b) / 2)
            + 1000 * max(offset + a - 1, 0)
        )

    @staticmethod
    def _params_from_fit(
        fit: GaussianParametersWithOffset,
    ) -> tuple[float, float, float]:
        return (fit.amplitude, fit.width, fit.offset)

    @staticmethod
    def _fit_from_params(
        *params: *tuple[float, ...],
    ) -> GaussianParametersWithOffset:
        return GaussianParametersWithOffset(
            params[0],
            np.abs(params[1]),
            params[2],
        )

    @staticmethod
    def _scale_params(
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, float, float]:
        return (params[0], dt * params[1], params[2])

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, 0, 0], [1, np.inf, 1])

    @classmethod
    def _fit_param_initial_guess(
        cls: type[Self],
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> tuple[float, float, float]:
        offset = 0.8 * np.min(np.abs(data["data"]))
        return (
            np.abs(data["data"][0]) - offset,
            get_free_particle_time(**info),
            offset,
        )

    def get_fit_from_isf(
        self: Self,
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> GaussianParametersWithOffset:
        # Stop trying to fit past the first non-decreasing ISF
        is_increasing = np.diff(np.abs(data["data"])) > 0
        first_increasing_idx = np.argmax(is_increasing).item()
        idx = data["basis"].n - 1 if first_increasing_idx == 0 else first_increasing_idx
        # Usually, near the cutoff point the curve becomes less gaussian
        idx -= min(10, idx // 2)
        idx = max(idx, 10)
        truncated = _truncate_value_list(data, idx)

        return super().get_fit_from_isf(truncated, **info)

    def get_rate_from_fit(
        self: Self,
        fit: GaussianParametersWithOffset,
    ) -> float:
        return 1 / fit.width

    def get_rate_label(self: Self) -> str:
        return "Gaussian (with offset)"

    def get_fit_times(
        self: Self,
        **info: Unpack[FitInfo],
    ) -> EvenlySpacedTimeBasis[Any, Any, Any]:
        return EvenlySpacedTimeBasis(100, 1, 0, 4 * get_free_particle_time(**info))


class DoubleGaussianMethod(FitMethod[tuple[GaussianParameters, GaussianParameters]]):
    """Fit the data to a double Gaussian."""

    def __init__(self: Self, ty: Literal["Fast", "Slow"]) -> None:  # noqa: D107
        self._ty = ty
        super().__init__()

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
    def _fit_param_initial_guess(
        cls: type[Self],
        data: ValueList[_BT0],
        **info: Unpack[FitInfo],
    ) -> tuple[float, float, float, float]:
        free_time = get_free_particle_time(**info)
        offset = 0.8 * np.min(np.abs(data["data"]))
        initial_height = (1 - offset) / 2
        return (initial_height, 0.5 * free_time, initial_height, 2 * free_time)

    @staticmethod
    def _params_from_fit(
        fit: tuple[GaussianParameters, GaussianParameters],
    ) -> tuple[float, float, float, float]:
        return (fit[0].amplitude, fit[0].width, fit[1].amplitude, fit[1].width)

    @staticmethod
    def _fit_from_params(
        *params: *tuple[float, ...],
    ) -> tuple[GaussianParameters, GaussianParameters]:
        return (
            GaussianParameters(params[0], np.abs(params[1])),
            GaussianParameters(params[2], np.abs(params[3])),
        )

    @staticmethod
    def _scale_params(
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        return (params[0], dt * params[1], params[2], dt * params[3])

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, 0, 0, 0], [1, np.inf, 1, np.inf])

    def get_rate_from_fit(
        self: Self,
        fit: tuple[GaussianParameters, GaussianParameters],
    ) -> float:
        return (
            max(1 / fit[0].width, 1 / fit[1].width)
            if self._ty == "Fast"
            else min(1 / fit[0].width, 1 / fit[1].width)
        )

    def get_rate_label(self: Self) -> str:
        return "Fast gaussian" if self._ty == "Fast" else "Slow gaussian"

    def get_fit_times(
        self: Self,
        **info: Unpack[FitInfo],
    ) -> EvenlySpacedTimeBasis[Any, Any, Any]:
        return EvenlySpacedTimeBasis(100, 1, 0, 4 * get_free_particle_time(**info))


@dataclass
class ExponentialParameters:
    """Parameters of an exponential fit."""

    amplitude: float
    time_constant: float


class ExponentialMethod(FitMethod[ExponentialParameters]):
    """Fit the data to an exponential."""

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
        *params: *tuple[float, ...],
    ) -> ExponentialParameters:
        return ExponentialParameters(params[0], params[1])

    @staticmethod
    def _scale_params(
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, float]:
        return (params[0], dt * params[1])

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, 0], [1, np.inf])

    @classmethod
    def _fit_param_initial_guess(
        cls: type[Self],
        data: ValueList[_BT0],  # noqa: ARG003
        **info: Unpack[FitInfo],
    ) -> tuple[float, float]:
        return (1, get_free_particle_time(**info))

    def get_rate_from_fit(
        self: Self,
        fit: ExponentialParameters,
    ) -> float:
        return 1 / fit.time_constant

    def get_rate_label(self: Self) -> str:
        return "Exponential"

    def get_fit_times(
        self: Self,
        **info: Unpack[FitInfo],
    ) -> EvenlySpacedTimeBasis[Any, Any, Any]:
        return EvenlySpacedTimeBasis(100, 1, 0, 40 * get_free_particle_time(**info))

    @classmethod
    def n_params(cls: type[Self]) -> int:
        return 2


class GaussianPlusExponentialMethod(
    FitMethod[tuple[GaussianParameters, ExponentialParameters]],
):
    """Fit the data to a gaussian plus an exponential."""

    def __init__(self: Self, ty: Literal["Gaussian", "Exponential"]) -> None:  # noqa: D107
        self._ty = ty
        super().__init__()

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
        *params: *tuple[float, ...],
    ) -> tuple[GaussianParameters, ExponentialParameters]:
        return (
            GaussianParameters(params[0], np.abs(params[1])),
            ExponentialParameters(params[2], params[3]),
        )

    @staticmethod
    def _scale_params(
        dt: float,
        params: tuple[float, ...],
    ) -> tuple[float, float, float, float]:
        return (params[0], dt * params[1], params[2], dt * params[3])

    @staticmethod
    def _fit_param_bounds() -> tuple[list[float], list[float]]:
        return ([0, 0, 0, 0], [1, np.inf, 1, np.inf])

    @classmethod
    def _fit_param_initial_guess(
        cls: type[Self],
        data: ValueList[_BT0],  # noqa: ARG003
        **info: Unpack[FitInfo],
    ) -> tuple[float, float, float, float]:
        free_time = get_free_particle_time(**info)
        return (0.5, free_time, 0.5, 2 * free_time)

    def get_rate_from_fit(
        self: Self,
        fit: tuple[GaussianParameters, ExponentialParameters],
    ) -> float:
        return 1 / fit[0].width if self._ty == "Gaussian" else 1 / fit[1].time_constant

    def get_rate_label(self: Self) -> str:
        return (
            "Gaussian (Gaussian + Exponential)"
            if self._ty == "Gaussian"
            else "Exponential  (Gaussian + Exponential)"
        )

    def get_fit_times(
        self: Self,
        **info: Unpack[FitInfo],
    ) -> EvenlySpacedTimeBasis[Any, Any, Any]:
        return EvenlySpacedTimeBasis(100, 1, 0, 40 * get_free_particle_time(**info))


def get_default_isf_times(
    *,
    include_negative: bool = False,
    **info: Unpack[FitInfo],
) -> EvenlySpacedTimeBasis[Any, Any, Any]:
    fit_time = GaussianMethod().get_fit_times(**info).delta_t
    if include_negative:
        return EvenlySpacedTimeBasis(101, 1, -50, fit_time)
    return EvenlySpacedTimeBasis(100, 1, 0, fit_time)
