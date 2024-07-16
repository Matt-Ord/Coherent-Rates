from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import Boltzmann, atomic_mass
from scipy.optimize import curve_fit
from surface_potential_analysis.basis.basis import FundamentalBasis
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operations import apply_operator_to_states
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    SingleBasisOperator,
    apply_operator_to_state,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products_elementwise,
)
from surface_potential_analysis.util.util import get_measured_data

from coherent_rates.system import get_hamiltonian, get_potential

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        StatisticalValueList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    from coherent_rates.system import PeriodicSystem, PeriodicSystemConfig

_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


@dataclass
class GaussianFitData:
    """Represents the parameters from a Gaussian fit."""

    amplitude: float
    amp_error: float
    sd: float
    sd_error: float


@dataclass
class DoubleGaussianFitData:
    """Represents the parameters from a double Gaussian fit."""

    amplitude1: float
    amp1_error: float
    sd1: float
    sd1_error: float
    amplitude2: float
    amp2_error: float
    sd2: float
    sd2_error: float


def _get_isf_pair_states_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    operator: SingleBasisOperator[Any],
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> tuple[StateVectorList[_AX0Inv, Any], StateVectorList[_AX0Inv, Any]]:
    state_evolved = solve_schrodinger_equation_diagonal(
        initial_state,
        times,
        hamiltonian,
    )
    state_evolved_scattered = apply_operator_to_states(operator, state_evolved)

    state_scattered = apply_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation_diagonal(
        state_scattered,
        times,
        hamiltonian,
    )
    return (state_evolved_scattered, state_scattered_evolved)


def get_isf_pair_states(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
    direction: tuple[int] = (1,),
) -> tuple[StateVectorList[_AX0Inv, Any], StateVectorList[_AX0Inv, Any]]:
    operator = get_periodic_x_operator(
        initial_state["basis"],
        direction=direction,
    )
    hamiltonian = get_hamiltonian(system, config)
    return _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )


def _get_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    operator: SingleBasisOperator[Any],
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> ValueList[_AX0Inv]:
    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )

    return calculate_inner_products_elementwise(
        state_scattered_evolved,
        state_evolved_scattered,
    )


def get_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
    direction: tuple[int] = (1,),
) -> ValueList[_AX0Inv]:
    operator = get_periodic_x_operator(
        initial_state["basis"],
        direction=direction,
    )
    hamiltonian = get_hamiltonian(system, config)

    return _get_isf_from_hamiltonian(hamiltonian, operator, initial_state, times)


def _get_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * temperature),
    )
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = boltzmann_distribution * np.exp(1j * phase) / normalization
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def _get_random_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    temperature: float,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    rng = np.random.default_rng()
    phase = 2 * np.pi * rng.random(len(hamiltonian["data"]))
    return _get_boltzmann_state_from_hamiltonian(hamiltonian, temperature, phase)


def get_random_boltzmann_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    """Generate a random Boltzmann state.

    Follows the formula described in eqn 5 in
    https://doi.org/10.48550/arXiv.2002.12035.


    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        temperature (float): temperature of the system

    Returns:
    -------
        StateVector[Any]: state with boltzmann distribution

    """
    hamiltonian = get_hamiltonian(system, config)
    return _get_random_boltzmann_state_from_hamiltonian(hamiltonian, config.temperature)


def _get_boltzmann_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    temperature: float,
    times: _AX0Inv,
    direction: tuple[int] = (1,),
    *,
    n_repeats: int = 1,
) -> ValueList[_AX0Inv]:
    isf_data = np.zeros(times.n, dtype=np.complex128)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = convert_operator_to_basis(
        get_periodic_x_operator(hamiltonian["basis"][0], direction),
        hamiltonian["basis"],
    )

    for _i in range(n_repeats):
        state = _get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            temperature,
        )
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data += data["data"]
    return {
        "data": isf_data / n_repeats,
        "basis": times,
    }


def get_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: _AX0Inv,
    direction: tuple[int] = (1,),
    *,
    n_repeats: int = 1,
) -> StatisticalValueList[_AX0Inv]:
    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    operator = get_periodic_x_operator(hamiltonian["basis"][0], direction)

    for i in range(n_repeats):
        state = _get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
        )
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[i, :] = data["data"]
    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": times,
        "standard_deviation": sd,
    }


def fit_abs_isf_to_gaussian(
    values: ValueList[_BT0],
) -> GaussianFitData:
    times = values["basis"]
    xdata1 = BasisUtil(times).nx_points

    ydata = get_measured_data(values["data"], "abs")

    def gauss(
        x: np.ndarray[Any, np.dtype[np.float64]],
        a: float,
        b: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return a * np.exp(-1 * np.square(x / b) / 2)

    parameters, covariance = curve_fit(gauss, xdata1, ydata)
    fit_A = parameters[0]
    fit_B = parameters[1]
    dt = times.times[1]

    return GaussianFitData(
        fit_A,
        np.sqrt(covariance[0][0]),
        fit_B * dt,
        np.sqrt(covariance[1][1]) * dt,
    )


def fit_abs_isf_to_double_gaussian(
    values: ValueList[_BT0],
) -> DoubleGaussianFitData:
    times = values["basis"]
    xdata1 = BasisUtil(times).nx_points

    ydata = get_measured_data(values["data"], "abs")

    def gauss(
        x: np.ndarray[Any, np.dtype[np.float64]],
        a1: float,
        b1: float,
        a2: float,
        b2: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return a1 * np.exp(-1 * np.square(x / b1) / 2) + a2 * np.exp(
            -1 * np.square(x / b2) / 2,
        )

    parameters, covariance = curve_fit(gauss, xdata1, ydata)
    fit_A1 = parameters[0]
    fit_B1 = parameters[1]
    fit_A2 = parameters[2]
    fit_B2 = parameters[3]
    dt = times.times[1]

    return DoubleGaussianFitData(
        fit_A1,
        np.sqrt(covariance[0][0]),
        fit_B1 * dt,
        np.sqrt(covariance[1][1]) * dt,
        fit_A2,
        np.sqrt(covariance[2][2]),
        fit_B2 * dt,
        np.sqrt(covariance[3][3]) * dt,
    )


def truncate_value_list(
    values: ValueList[EvenlySpacedTimeBasis[int, int, int]],
    index: int,
) -> ValueList[EvenlySpacedTimeBasis[int, int, int]]:
    data = values["data"][0 : index + 1]
    new_times = EvenlySpacedTimeBasis(index + 1, 1, 0, values["basis"].times[index])
    return {
        "basis": new_times,
        "data": data,
    }


class MomentumBasis(FundamentalBasis[Any]):
    def __init__(self, k_points: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self._k_points = k_points
        super().__init__(k_points.size)  # type: ignore

    @property
    def k_points(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._k_points


def get_ak_data_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    k_points: list[int] | None = None,
) -> ValueList[MomentumBasis]:
    mass_ratio = system.mass / atomic_mass
    times = EvenlySpacedTimeBasis(100, 1, 0, 1.5e-14 * np.power(mass_ratio, 0.6))
    length = get_potential(system, config)["basis"].delta_x_stacked[0]
    if k_points is None:
        xdata = (np.arange(int(config.shape[0] / 2)) * 2 + 1) * int(
            config.resolution[0] / 2,
        )
    else:
        xdata = k_points

    ydata = np.zeros(len(xdata))
    hamiltonian = get_hamiltonian(system, config)
    for i in range(len(xdata)):
        isf = _get_boltzmann_isf_from_hamiltonian(
            hamiltonian,
            config.temperature,
            times,
            (xdata[i],),
            n_repeats=10,
        )

        data = np.abs(isf["data"])

        diff = np.diff(data)
        positive_diff = diff > 0
        index = np.argmax(positive_diff)
        idx = times.n - 1 if positive_diff[index] == 0 else index

        truncated_isf = truncate_value_list(isf, idx)
        ydata[i] = 1 / fit_abs_isf_to_gaussian(truncated_isf).sd
        times = EvenlySpacedTimeBasis(
            times.n,
            times.step,
            times.offset,
            times.times[idx],
        )
    xdata_scaled = np.array(xdata) * 2 * np.pi / length
    basis = MomentumBasis(xdata_scaled)
    return {
        "data": ydata,
        "basis": basis,
    }


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])
