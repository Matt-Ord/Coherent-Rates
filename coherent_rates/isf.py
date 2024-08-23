from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import numpy as np
from scipy.constants import Boltzmann  # type: ignore library type
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
)
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
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
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_position_basis,
)
from surface_potential_analysis.state_vector.eigenstate_calculation import (
    calculate_expectation_diagonal,
)
from surface_potential_analysis.state_vector.eigenstate_list import (
    ValueList,
)
from surface_potential_analysis.state_vector.plot import (
    get_periodic_x_operator,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products_elementwise,
)
from surface_potential_analysis.util.decorators import npy_cached_dict, timed
from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

from coherent_rates.fit import FitMethod, GaussianMethod, GaussianPlusExponentialMethod
from coherent_rates.scattering_operator import (
    SparseScatteringOperator,
    apply_scattering_operator_to_state,
    apply_scattering_operator_to_states,
    get_periodic_x_operator_sparse,
)
from coherent_rates.system import (
    get_coherent_state,
    get_hamiltonian,
    get_thermal_occupation_k,
    get_thermal_occupation_x,
)

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

_BV0 = TypeVar("_BV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])

_ESB0 = TypeVar("_ESB0", bound=BlochBasis[Any])
_ESB1 = TypeVar("_ESB1", bound=BlochBasis[Any])


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
    StateVectorList[_BT0, StackedBasisWithVolumeLike[Any, Any, Any]],
    StateVectorList[_BT0, StackedBasisWithVolumeLike[Any, Any, Any]],
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
        BlochBasis[_B0],
    ],
) -> StateVectorList[
    TupleBasis[_B0, _B1],
    TupleBasisLike[*tuple[FundamentalTransformedBasis[Any], ...]],
]:
    basis = states["basis"][1].wavefunctions["basis"][0]

    data = states["data"].reshape(-1, *basis.shape).swapaxes(0, 1)
    return {
        "basis": TupleBasis(TupleBasis(basis[0], states["basis"][0]), basis[1]),
        "data": data.ravel(),
    }


def _get_band_resolved_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_ESB1],
    operator: SparseScatteringOperator[_ESB0, _ESB0],
    initial_state: StateVector[_B1],
    times: _BT0,
) -> ValueList[TupleBasis[Any, _BT0]]:
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


def get_random_coherent_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    sigma_0: float,
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    """Generate a Gaussian state with x0,k0 given approximately by a thermal distribution.

    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        sigma_0 (float): width of the state

    Returns:
    -------
        StateVector[...]: random coherent state

    """
    potential = convert_potential_to_position_basis(
        system.get_potential(config.shape, config.resolution),
    )
    basis = potential["basis"]
    util = BasisUtil(basis)

    # position probabilities
    x_probability_normalized = get_thermal_occupation_x(system, config)
    x_index = np.random.choice(util.nx_points, p=x_probability_normalized)
    x0 = util.get_stacked_index(x_index)

    # momentum probabilities
    k_probability_normalized = get_thermal_occupation_k(system, config)
    k_index = np.random.choice(util.nx_points, p=k_probability_normalized)
    k0 = util.get_stacked_index(k_index)

    return get_coherent_state(basis, x0, k0, sigma_0)


@timed
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
    bands = hamiltonian["basis"][0].wavefunctions["basis"][0][0]
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


def get_coherent_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: _BT0,
    *,
    direction: tuple[int, ...] | None = None,
    n_repeats: int = 10,
    sigma_0: float | None = None,
) -> StatisticalValueList[_BT0]:
    sigma_0 = system.lattice_constant / 10 if sigma_0 is None else sigma_0
    hamiltonian = get_hamiltonian(system, config)
    operator = get_periodic_x_operator_sparse(
        hamiltonian["basis"][1],
        direction=direction,
    )

    isf_data = np.zeros((n_repeats, times.n), dtype=np.complex128)
    for i in range(n_repeats):
        state = get_random_coherent_state(system, config, sigma_0)
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[i, :] = data["data"]

    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": times,
        "standard_deviation": sd,
    }


class MomentumBasis(FundamentalBasis[Any]):  # noqa: D101
    def __init__(self, k_points: np.ndarray[Any, np.dtype[np.float64]]) -> None:  # noqa: D107, ANN101
        self._k_points = k_points
        super().__init__(k_points.size)

    @property
    def k_points(self: Self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._k_points


def get_free_particle_time(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    direction: tuple[int, ...] | None = None,
) -> float:
    direction = tuple(1 for _ in config.shape) if direction is None else direction
    basis = system.get_potential(config.shape, config.resolution)["basis"]
    dk_stacked = BasisUtil(basis).dk_stacked

    k = np.linalg.norm(np.einsum("i,ij->j", direction, dk_stacked))  # type: ignore library type
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


def _get_rate_against_momentum_data_path(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    nk_points: list[tuple[int, ...]] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,  # noqa: ARG001
) -> Path:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    return Path(
        f"data/{hash((system, config))}.{hash(nk_points[0])}.{hash(fit_method)}.npz",
    )


def get_value_list_at_idx(
    values: ValueList[TupleBasis[_B0, MomentumBasis]],
    index: int,
) -> ValueList[MomentumBasis]:
    basis = values["basis"][1]
    full_data = values["data"].reshape((values["basis"][0].n, values["basis"][1].n))
    data = full_data[index, :]
    return {"basis": basis, "data": data}


@npy_cached_dict(_get_rate_against_momentum_data_path, load_pickle=True)
def get_rate_against_momentum_data(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    nk_points: list[tuple[int, ...]] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
) -> ValueList[TupleBasis[FundamentalBasis[int], MomentumBasis]]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points

    rates = np.zeros((fit_method.n_rates(), len(nk_points)), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    for i, direction in enumerate(nk_points):
        free_times = (
            EvenlySpacedTimeBasis(
                100,
                1,
                0,
                fit_method.get_fit_time(system, config, direction),
            )
            if times is None
            else times
        )

        isf = _get_boltzmann_isf_from_hamiltonian(
            hamiltonian,
            config.temperature,
            free_times,
            direction,
            n_repeats=10,
        )

        rates[:, i] = fit_method.get_rates_from_isf(isf)

    dk_stacked = BasisUtil(hamiltonian["basis"][0]).dk_stacked
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)  # type: ignore library type
    basis = MomentumBasis(k_points)
    return {
        "data": rates.ravel(),
        "basis": TupleBasis(FundamentalBasis(fit_method.n_rates()), basis),
    }


def _get_scattered_energy_change(
    hamiltonian: SingleBasisDiagonalOperator[_BV0],
    state: StateVector[Any],
    direction: tuple[int, ...] | None = None,
) -> float:
    operator = get_periodic_x_operator(hamiltonian["basis"][0], direction)

    energy = np.real(calculate_expectation_diagonal(hamiltonian, state))
    scattered_state = apply_operator_to_state(operator, state)
    scattered_energy = calculate_expectation_diagonal(
        hamiltonian,
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
) -> StatisticalValueList[MomentumBasis]:
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    hamiltonian = get_hamiltonian(system, config)
    energy_change = np.zeros(len(nk_points), dtype=np.complex128)
    standard_deviation = np.zeros(len(nk_points), dtype=np.float64)
    for i, k_point in enumerate(nk_points):
        data = _get_thermal_scattered_energy_change(
            hamiltonian,
            config.temperature,
            k_point,
            n_repeats=n_repeats,
        )["data"]
        energy_change[i] = np.average(data)
        standard_deviation[i] = np.std(data)

    dk_stacked = BasisUtil(hamiltonian["basis"][0]).dk_stacked
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)  # type: ignore library type
    basis = MomentumBasis(k_points)
    return {
        "data": energy_change,
        "basis": basis,
        "standard_deviation": standard_deviation,
    }


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
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)  # type: ignore library type
    basis = MomentumBasis(k_points)
    return {"data": energy_change, "basis": basis}


def calculate_effective_mass_from_gradient(
    config: PeriodicSystemConfig,
    gradient: float,
) -> float:
    return Boltzmann * config.temperature / (gradient * gradient)


@dataclass
class RateAgainstMomentumFitData:
    """_Stores data from linear fit with calculated effective mass."""

    gradient: float
    intercept: float


def get_rate_against_momentum_linear_fit(
    values: ValueList[MomentumBasis],
) -> RateAgainstMomentumFitData:
    k_points = values["basis"].k_points
    rates = values["data"]
    gradient, intercept = np.polyfit(k_points, rates, 1)
    return RateAgainstMomentumFitData(gradient, intercept)


@timed
def get_rate_against_temperature_and_momentum_data(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    temperatures: list[int] | None = None,
    nk_points: list[tuple[int, ...]] | None = None,
) -> ValueList[
    TupleBasis[TupleBasis[FundamentalBasis[int], FundamentalBasis[int]], MomentumBasis]
]:
    fit_method = GaussianPlusExponentialMethod() if fit_method is None else fit_method
    nk_points = _get_default_nk_points(config) if nk_points is None else nk_points
    temperatures = (
        [(60 + 30 * i) for i in range(5)] if temperatures is None else temperatures
    )

    n_rates = fit_method.n_rates()
    n_temperatures = len(temperatures)
    data = np.zeros(
        (n_rates, n_temperatures, len(nk_points)),
        dtype=np.complex128,
    )

    for j, temperature in enumerate(temperatures):
        config.temperature = temperature
        rate_data = get_rate_against_momentum_data(
            system,
            config,
            fit_method=fit_method,
            nk_points=nk_points,
        )["data"]
        rate_data = rate_data.reshape(n_rates, -1)
        data[:, j, :] = rate_data

    hamiltonian = get_hamiltonian(system, config)
    dk_stacked = BasisUtil(hamiltonian["basis"][0]).dk_stacked
    k_points = np.linalg.norm(np.einsum("ij,jk->ik", nk_points, dk_stacked), axis=1)  # type: ignore einsum
    basis = MomentumBasis(k_points)

    return {
        "data": data.ravel(),
        "basis": TupleBasis(
            TupleBasis(
                FundamentalBasis(n_rates),
                FundamentalBasis(n_temperatures),
            ),
            basis,
        ),
    }
