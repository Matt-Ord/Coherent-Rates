from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import numpy as np
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
)
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
)
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    apply_operator_to_state,
    as_operator,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_expectation,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products_elementwise,
)
from surface_potential_analysis.util.decorators import npy_cached_dict, timed
from surface_potential_analysis.util.util import get_measured_data
from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

from coherent_rates.scattering_operator import (
    SparseScatteringOperator,
    apply_scattering_operator_to_state,
    apply_scattering_operator_to_states,
    get_periodic_x_operator_sparse,
)
from coherent_rates.system import get_hamiltonian

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        TupleBasisLike,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
    )
    from surface_potential_analysis.state_vector.eigenstate_list import (
        StatisticalValueList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    from coherent_rates.system import PeriodicSystem, PeriodicSystemConfig

_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])

_B0 = TypeVar("_B0", bound=BasisLike[Any, Any])
_B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
_B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

_BV0 = TypeVar("_BV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])

_ESB0 = TypeVar("_ESB0", bound=BlochBasis[Any])


def _get_isf_pair_states_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> tuple[StateVectorList[_BT0, _ESB0], StateVectorList[_BT0, _B0]]:
    state_evolved = solve_schrodinger_equation_diagonal(
        initial_state,
        times,
        hamiltonian,
    )

    state_evolved_scattered = apply_scattering_operator_to_states(
        operator,
        state_evolved,
    )

    state_scattered = apply_scattering_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation_diagonal(
        state_scattered,
        times,
        hamiltonian,
    )

    return (state_evolved_scattered, state_scattered_evolved)


def get_isf_pair_states(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_B1],
    times: _BT0,
    direction: tuple[int, ...] | None = None,
) -> tuple[
    StateVectorList[_BT0, BasisLike[Any, Any]],
    StateVectorList[_BT0, BasisLike[Any, Any]],
]:
    hamiltonian = get_hamiltonian(system, config)
    operator = get_periodic_x_operator_sparse(
        hamiltonian["basis"][1],
        direction=direction,
    )

    return _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )


@timed
def _get_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> ValueList[_BT0]:
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


def _get_states_per_band(
    states: StateVectorList[
        _B1,
        ExplicitStackedBasisWithLength[TupleBasisLike[_B0, _B2], Any],
    ],
) -> StateVectorList[TupleBasis[_B0, _B1], _B2]:
    basis = states["basis"][1].vectors["basis"][0]

    data = states["data"].reshape(-1, *basis.shape).swapaxes(0, 1)
    return {
        "basis": TupleBasis(TupleBasis(basis[0], states["basis"][0]), basis[1]),
        "data": data.ravel(),
    }


def _get_band_resolved_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[
        ExplicitStackedBasisWithLength[TupleBasisLike[_B0, Any], Any]
    ],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> ValueList[TupleBasis[_B0, _BT0]]:
    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )
    per_band_scattered_evolved = _get_states_per_band(
        state_scattered_evolved,
    )
    per_band_evolved_scattered = _get_states_per_band(
        state_evolved_scattered,
    )

    return calculate_inner_products_elementwise(
        per_band_scattered_evolved,
        per_band_evolved_scattered,
    )


def get_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_B1],
    times: _BT0,
    direction: tuple[int, ...] | None = None,
) -> ValueList[_BT0]:
    hamiltonian = get_hamiltonian(system, config)
    operator = get_periodic_x_operator_sparse(
        hamiltonian["basis"][1],
        direction=direction,
    )

    return _get_isf_from_hamiltonian(hamiltonian, operator, initial_state, times)


def _get_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> StateVector[_B0]:
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * temperature),
    )
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = boltzmann_distribution * np.exp(1j * phase) / normalization
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def _get_random_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    temperature: float,
) -> StateVector[_B0]:
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
    hamiltonian: SingleBasisDiagonalOperator[_ESB0],
    temperature: float,
    times: _BT0,
    direction: tuple[int, ...] | None = None,
    *,
    n_repeats: int = 1,
) -> StatisticalValueList[_BT0]:
    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = get_periodic_x_operator_sparse(
        hamiltonian["basis"][1],
        direction=direction,
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
    times: _BT0,
    direction: tuple[int, ...] | None = None,
    *,
    n_repeats: int = 1,
) -> StatisticalValueList[_BT0]:
    direction = tuple(1 for _ in config.shape) if direction is None else direction
    hamiltonian = get_hamiltonian(system, config)
    return _get_boltzmann_isf_from_hamiltonian(
        hamiltonian,
        config.temperature,
        times,
        direction,
        n_repeats=n_repeats,
    )


def get_band_resolved_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: _BT0,
    direction: tuple[int, ...] | None = None,
    *,
    n_repeats: int = 1,
) -> StatisticalValueList[TupleBasisLike[BasisLike[Any, Any], _BT0]]:
    hamiltonian = get_hamiltonian(system, config)  #
    bands = hamiltonian["basis"][0].vectors["basis"][0][0]
    operator = get_periodic_x_operator_sparse(hamiltonian["basis"][0], direction)

    isf_data = np.zeros((n_repeats, bands.n * times.n), dtype=np.complex128)

    for i in range(n_repeats):
        state = _get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
        )
        data = _get_band_resolved_isf_from_hamiltonian(
            hamiltonian,
            operator,
            state,
            times,
        )
        isf_data[i, :] = data["data"]

    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": TupleBasis(bands, times),
        "standard_deviation": sd,
    }


@dataclass
class GaussianFitData:
    """Represents the parameters from a Gaussian fit."""

    amplitude: float
    amplitude_error: float
    width: float
    width_error: float


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


class MomentumBasis(FundamentalBasis[Any]):  # noqa: D101
    def __init__(self, k_points: np.ndarray[Any, np.dtype[np.float64]]) -> None:  # noqa: D107, ANN101
        self._k_points = k_points
        super().__init__(k_points.size)

    @property
    def k_points(self: Self) -> np.ndarray[Any, np.dtype[np.float64]]:  # noqa: D102
        return self._k_points


def get_free_particle_time(
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


def _get_default_nk_points(config: PeriodicSystemConfig) -> list[tuple[int, ...]]:
    return list(
        zip(
            *tuple(
                cast(list[int], (s * np.arange(1, r)).tolist())
                for (s, r) in zip(config.shape, config.resolution, strict=True)
            ),
        ),
    )


def _get_ak_data_path(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[tuple[int, ...]] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,  # noqa: ARG001
) -> Path:
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    return Path(f"data/{hash((system, config))}.{hash(nk_points[0])}.npz")


@npy_cached_dict(_get_ak_data_path, load_pickle=True)
def get_ak_data(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[tuple[int, ...]] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
) -> ValueList[MomentumBasis]:
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    free_time = get_free_particle_time(system, config, nk_points[0])
    times = (
        EvenlySpacedTimeBasis(
            100,
            1,
            0,
            4 * free_time,
        )
        if times is None
        else times
    )

    rates = np.zeros(len(nk_points), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i, direction in enumerate(nk_points):
        isf = _get_boltzmann_isf_from_hamiltonian(
            hamiltonian,
            config.temperature,
            times,
            direction,
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
    dk_stacked = BasisUtil(hamiltonian["basis"][0]).dk_stacked
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)
    basis = MomentumBasis(k_points)
    return {"data": rates, "basis": basis}


def _get_scattered_energy_change(
    hamiltonian: SingleBasisDiagonalOperator[_BV0],
    state: StateVector[Any],
    direction: tuple[int, ...] | None = None,
) -> float:
    hamiltonian_operator = as_operator(hamiltonian)
    operator = get_periodic_x_operator(hamiltonian["basis"][0], direction)

    energy = np.real(calculate_expectation(hamiltonian_operator, state))
    scattered_state = apply_operator_to_state(operator, state)
    scattered_energy = calculate_expectation(
        hamiltonian_operator,
        scattered_state,
    )

    return np.real(scattered_energy - energy)


def _get_thermal_scattered_energy_change(
    hamiltonian: SingleBasisDiagonalOperator[_BV0],
    temperature: float,
    direction: tuple[int, ...] | None = None,
    *,
    n_repeats: int = 10,
) -> ValueList[FundamentalBasis[int]]:
    energy_change = np.zeros(n_repeats, dtype=np.complex128)
    for i in range(n_repeats):
        state = _get_random_boltzmann_state_from_hamiltonian(hamiltonian, temperature)
        energy_change[i] = _get_scattered_energy_change(hamiltonian, state, direction)

    return {"basis": FundamentalBasis(n_repeats), "data": energy_change}


def get_thermal_scattered_energy_change_against_k(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
) -> ValueList[MomentumBasis]:
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    hamiltonian = get_hamiltonian(system, config)
    energy_change = np.zeros(len(nk_points), dtype=np.complex128)
    for i, k_point in enumerate(nk_points):
        energy_change[i] = np.average(
            _get_thermal_scattered_energy_change(
                hamiltonian,
                config.temperature,
                k_point,
                n_repeats=n_repeats,
            )["data"],
        )

    dk_stacked = BasisUtil(hamiltonian["basis"][0]).dk_stacked
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)
    basis = MomentumBasis(k_points)
    return {"data": energy_change, "basis": basis}


def get_scattered_energy_change_against_k(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    state: StateVector[Any],
    *,
    nk_points: list[tuple[int, ...]] | None = None,
) -> ValueList[MomentumBasis]:
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    hamiltonian = get_hamiltonian(system, config)
    energy_change = np.zeros(len(nk_points), dtype=np.complex128)
    for i, k_point in enumerate(nk_points):
        energy_change[i] = _get_scattered_energy_change(hamiltonian, state, k_point)

    dk_stacked = BasisUtil(hamiltonian["basis"][0]).dk_stacked
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)
    basis = MomentumBasis(k_points)
    return {"data": energy_change, "basis": basis}


def calculate_effective_mass_from_gradient(
    config: PeriodicSystemConfig,
    gradient: float,
) -> float:
    return Boltzmann * config.temperature / (gradient * gradient)


@dataclass
class AlphaDeltakFitData:
    """_Stores data from linear fit with calculated effective mass."""

    gradient: float
    intercept: float
    effective_mass: float


def get_alpha_deltak_linear_fit(
    config: PeriodicSystemConfig,
    values: ValueList[MomentumBasis],
) -> AlphaDeltakFitData:
    k_points = values["basis"].k_points
    rates = values["data"]
    gradient, intercept = np.polyfit(k_points, rates, 1)
    effective_mass = calculate_effective_mass_from_gradient(config, gradient)
    return AlphaDeltakFitData(gradient, intercept, effective_mass)
