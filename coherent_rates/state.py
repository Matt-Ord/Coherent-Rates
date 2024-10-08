from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

import numpy as np
from scipy.constants import (  # type: ignore bad types
    Boltzmann,
    hbar,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
    get_displacements_x_stacked,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_position_basis,
    get_continuous_potential,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)

from coherent_rates.solve import get_hamiltonian

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalPositionBasis,
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.basis_like import (
        BasisLike,
    )
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList
    from surface_potential_analysis.state_vector.state_vector import StateVector

    from coherent_rates.config import PeriodicSystemConfig
    from coherent_rates.system import System

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def get_coherent_state(
    basis: _SBV0,
    x_0: tuple[float, ...],
    k_0: tuple[float, ...],
    sigma_0: tuple[float, ...],
) -> StateVector[_SBV0]:
    basis_x = stacked_basis_as_fundamental_position_basis(basis)

    displacements = get_displacements_x_stacked(basis, x_0)

    # stores distance from x0
    distance = np.linalg.norm(
        [d["data"] / s for d, s in zip(displacements, sigma_0)],
        axis=0,
    )

    # i k.(x - x')
    phi = (2 * np.pi) * np.einsum(  # type: ignore unknown lib type
        "ij,i->j",
        [d["data"] for d in displacements],
        k_0,
    )
    data = np.exp(-1j * phi - np.square(distance) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return convert_state_vector_to_basis({"basis": basis_x, "data": data / norm}, basis)


@overload
def get_thermal_probability_x(
    system: System,
    config: PeriodicSystemConfig,
    x_point: tuple[float, ...],
) -> float:
    ...


@overload
def get_thermal_probability_x(
    system: System,
    config: PeriodicSystemConfig,
    x_point: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ...


def get_thermal_probability_x(
    system: System,
    config: PeriodicSystemConfig,
    x_point: tuple[float, ...] | tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    potential = get_continuous_potential(
        system.get_potential(config.shape, config.resolution),
    )
    return np.abs(
        np.exp(-potential(cast(Any, x_point)) / (config.temperature * Boltzmann)),
    )


def get_thermal_occupation_x(
    system: System,
    config: PeriodicSystemConfig,
) -> ValueList[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    potential = convert_potential_to_position_basis(
        system.get_potential(config.shape, config.resolution),
    )
    x_probability = get_thermal_probability_x(
        system,
        config,
        tuple(BasisUtil(potential["basis"]).x_points_stacked),
    )
    return {"basis": potential["basis"], "data": x_probability / np.sum(x_probability)}


@overload
def get_thermal_probability_k(
    system: System,
    config: PeriodicSystemConfig,
    k_point: tuple[float, ...],
) -> float:
    ...


@overload
def get_thermal_probability_k(
    system: System,
    config: PeriodicSystemConfig,
    k_point: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    ...


def get_thermal_probability_k(
    system: System,
    config: PeriodicSystemConfig,
    k_point: tuple[float, ...] | tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> float | np.ndarray[Any, np.dtype[np.float64]]:
    return np.abs(
        np.exp(
            -np.square(hbar * np.linalg.norm(k_point, axis=0))
            / (2 * system.mass * config.temperature * Boltzmann),
        ),
    )


def get_thermal_occupation_k(
    system: System,
    config: PeriodicSystemConfig,
) -> ValueList[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    basis = system.get_potential_basis(config.shape, config.resolution)
    k_basis = stacked_basis_as_fundamental_momentum_basis(basis)
    util = BasisUtil(k_basis)
    k_probability = get_thermal_probability_k(
        system,
        config,
        tuple(util.fundamental_stacked_k_points),
    )
    return {"basis": k_basis, "data": k_probability / np.sum(k_probability)}


def get_random_coherent_x(
    system: System,
    config: PeriodicSystemConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, ...]:
    rng = np.random.default_rng() if rng is None else rng

    basis = stacked_basis_as_fundamental_position_basis(
        system.get_potential_basis(config.shape, config.resolution),
    )
    util = BasisUtil(basis)

    while True:
        x0 = tuple[float, ...](
            np.einsum("i,ik->k", rng.random(basis.ndim), util.delta_x_stacked),  # type: ignore lib
        )
        if rng.random() > get_thermal_probability_x(system, config, x0):
            continue
        return x0


def get_random_coherent_k(
    system: System,
    config: PeriodicSystemConfig,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[float, ...]:
    rng = np.random.default_rng() if rng is None else rng

    basis = stacked_basis_as_fundamental_position_basis(
        system.get_potential_basis(config.shape, config.resolution),
    )
    util = BasisUtil(basis)

    while True:
        x0 = tuple[float, ...](
            np.einsum("i,ik->k", (0.5 - rng.random(basis.ndim)), util.delta_k_stacked),  # type: ignore lib
        )
        if rng.random() > get_thermal_probability_k(system, config, x0):
            continue
        return x0


def get_random_coherent_coordinates(
    system: System,
    config: PeriodicSystemConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rng = np.random.default_rng()

    # position probabilities
    x0 = get_random_coherent_x(
        system,
        config,
        rng=rng,
    )

    # momentum probabilities
    k0 = get_random_coherent_k(
        system,
        config,
        rng=rng,
    )

    return (x0, k0)


def get_random_coherent_state(
    system: System,
    config: PeriodicSystemConfig,
    sigma_0: tuple[float, ...],
) -> StateVector[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[Any, Any], ...]]
]:
    """Generate a Gaussian state.

    x0,k0 are given approximately by a thermal distribution.

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

    x0, k0 = get_random_coherent_coordinates(system, config)

    return get_coherent_state(basis, x0, k0, sigma_0)


def get_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
) -> StateVector[_B0]:
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * temperature),
    )
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = (
        boltzmann_distribution / normalization
        if phase is None
        else boltzmann_distribution * np.exp(1j * phase) / normalization
    )
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def get_random_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    temperature: float,
) -> StateVector[_B0]:
    rng = np.random.default_rng()
    phase = 2 * np.pi * rng.random(len(hamiltonian["data"]))
    return get_boltzmann_state_from_hamiltonian(hamiltonian, temperature, phase)


def get_random_boltzmann_state(
    system: System,
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
    return get_random_boltzmann_state_from_hamiltonian(hamiltonian, config.temperature)
