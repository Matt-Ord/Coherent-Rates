from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

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

from coherent_rates.system import get_hamiltonian

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
) -> StatisticalValueList[_AX0Inv]:
    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = convert_operator_to_basis(
        get_periodic_x_operator(hamiltonian["basis"][0], direction),
        hamiltonian["basis"],
    )
    for i in range(n_repeats):
        state = _get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            temperature,
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


@dataclass
class GaussianFitData:
    """Represents the parameters from a Gaussian fit."""

    amplitude: float
    amplitude_error: float
    width: float
    width_error: float


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])


def fit_abs_isf_to_gaussian(
    values: ValueList[_BT0],
) -> GaussianFitData:
    def gaussian(
        x: np.ndarray[Any, np.dtype[np.float64]],
        a: float,
        b: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return a * np.exp(-1 * np.square(x / b) / 2)

    x_data = BasisUtil(values["basis"]).nx_points
    y_data = get_measured_data(values["data"], "abs")
    parameters, covariance = curve_fit(gaussian, x_data, y_data)
    fit_A = parameters[0]
    fit_B = parameters[1]
    dt = values["basis"].times[1]

    return GaussianFitData(
        fit_A,
        np.sqrt(covariance[0][0]),
        fit_B * dt,
        np.sqrt(covariance[1][1]) * dt,
    )


def fit_abs_isf_to_double_gaussian(
    values: ValueList[_BT0],
) -> tuple[GaussianFitData, GaussianFitData]:
    def double_gaussian(
        x: np.ndarray[Any, np.dtype[np.float64]],
        a1: float,
        b1: float,
        a2: float,
        b2: float,
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        return a1 * np.exp(-1 * np.square(x / b1) / 2) + a2 * np.exp(
            -1 * np.square(x / b2) / 2,
        )

    x_data = BasisUtil(values["basis"]).nx_points
    y_data = get_measured_data(values["data"], "abs")
    parameters, covariance = curve_fit(double_gaussian, x_data, y_data)
    fit_A1 = parameters[0]
    fit_B1 = parameters[1]
    fit_A2 = parameters[2]
    fit_B2 = parameters[3]
    dt = values["basis"].times[1]

    return (
        GaussianFitData(
            fit_A1,
            np.sqrt(covariance[0][0]),
            fit_B1 * dt,
            np.sqrt(covariance[1][1]) * dt,
        ),
        GaussianFitData(
            fit_A2,
            np.sqrt(covariance[2][2]),
            fit_B2 * dt,
            np.sqrt(covariance[3][3]) * dt,
        ),
    )


def truncate_value_list(
    values: ValueList[EvenlySpacedTimeBasis[int, int, int]],
    index: int,
) -> ValueList[EvenlySpacedTimeBasis[int, int, int]]:
    data = values["data"][0 : index + 1]
    new_times = EvenlySpacedTimeBasis(index + 1, 1, 0, values["basis"].times[index])
    return {"basis": new_times, "data": data}


class MomentumBasis(FundamentalBasis[Any]):
    def __init__(self, k_points: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self._k_points = k_points
        super().__init__(k_points.size)

    @property
    def k_points(self: Self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._k_points


def get_ak_data_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[int] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
) -> ValueList[MomentumBasis]:
    mass_ratio = system.mass / atomic_mass
    times = (
        EvenlySpacedTimeBasis(100, 1, 0, 1.5e-14 * np.power(mass_ratio, 0.6))
        if times is None
        else times
    )
    nk_points = (
        cast(list[int], (config.shape[0] * np.arange(1, config.resolution[0])).tolist())
        if nk_points is None
        else nk_points
    )
    rates = np.zeros(len(nk_points), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i in range(len(nk_points)):
        isf = _get_boltzmann_isf_from_hamiltonian(
            hamiltonian,
            config.temperature,
            times,
            (nk_points[i],),
            n_repeats=10,
        )

        is_increasing = np.diff(np.abs(isf["data"])) > 0
        first_increasing_idx = np.argmax(is_increasing).item()
        idx = times.n - 1 if first_increasing_idx == 0 else first_increasing_idx

        truncated_isf = truncate_value_list(isf, idx)
        rates[i] = 1 / fit_abs_isf_to_gaussian(truncated_isf).width
        times = EvenlySpacedTimeBasis(
            times.n,
            times.step,
            times.offset,
            times.times[idx],
        )
    k_points = np.array(nk_points) * BasisUtil(hamiltonian["basis"][0]).dk_stacked[0]
    basis = MomentumBasis(k_points)
    return {"data": rates, "basis": basis}
