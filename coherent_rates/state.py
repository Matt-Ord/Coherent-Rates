from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
from scipy.constants import (  # type: ignore bad types
    Boltzmann,
    hbar,
)
from surface_potential_analysis.basis.util import (
    BasisUtil,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_position_basis,
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
    )
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import SingleBasisDiagonalOperator
    from surface_potential_analysis.state_vector.state_vector import StateVector

    from coherent_rates.config import PeriodicSystemConfig
    from coherent_rates.system import PeriodicSystem

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])
    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])


def get_coherent_state(
    basis: _SBV0,
    x_0: tuple[int, ...],
    k_0: tuple[int, ...],
    sigma_0: float,
) -> StateVector[_SBV0]:
    basis_x = stacked_basis_as_fundamental_position_basis(basis)

    util = BasisUtil(basis_x)
    dx_stacked = util.dx_stacked

    idx = util.get_flat_index(x_0)

    # nx[i,j] stores the ith component displacement jth point from x0
    nx = tuple(
        (n_x_points[idx] - n_x_points[:] + n // 2) % n - (n // 2)
        for (n_x_points, n) in zip(
            util.fundamental_stacked_nx_points,
            util.fundamental_shape,
            strict=True,
        )
    )
    # stores distance from x0
    distance = np.linalg.norm(np.einsum("ji,jk->ik", nx, dx_stacked), axis=1)  # type: ignore unknown

    # i k.(x - x')
    dk = tuple(n / f for (n, f) in zip(k_0, basis_x.shape))
    phi = (2 * np.pi) * np.einsum(  # type: ignore unknown lib type
        "ij,i->j",
        nx,
        dk,
    )
    data = np.exp(-1j * phi - np.square(distance / sigma_0) / 2)
    norm = np.sqrt(np.sum(np.square(np.abs(data))))

    return convert_state_vector_to_basis({"basis": basis_x, "data": data / norm}, basis)


def get_thermal_occupation_x(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    potential = convert_potential_to_position_basis(
        system.get_potential(config.shape, config.resolution),
    )
    x_probability = np.abs(
        np.exp(-potential["data"] / (config.temperature * Boltzmann)),
    )
    return x_probability / np.sum(x_probability)


def get_thermal_occupation_k(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    basis = system.get_potential_basis(config.shape, config.resolution)
    k_basis = stacked_basis_as_fundamental_momentum_basis(basis)
    util = BasisUtil(k_basis)
    k_distance = np.linalg.norm(util.fundamental_stacked_k_points, axis=0)
    k_probability = np.abs(
        np.exp(
            -np.square(hbar * k_distance)
            / (2 * system.mass * config.temperature * Boltzmann),
        ),
    )
    return k_probability / np.sum(k_probability)


def get_random_coherent_coordinates(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    basis = stacked_basis_as_fundamental_position_basis(
        system.get_potential_basis(config.shape, config.resolution),
    )
    util = BasisUtil(basis)

    rng = np.random.default_rng()

    # position probabilities
    x_probability_normalized = get_thermal_occupation_x(system, config)
    x_index = rng.choice(util.nx_points, p=x_probability_normalized)
    x0 = cast(tuple[int, ...], util.get_stacked_index(x_index))

    # momentum probabilities
    k_probability_normalized = get_thermal_occupation_k(system, config)
    k_index = rng.choice(util.nx_points, p=k_probability_normalized)
    k0 = cast(tuple[int, ...], util.get_stacked_index(k_index))
    return (x0, k0)


def get_random_coherent_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    sigma_0: float,
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
    return get_random_boltzmann_state_from_hamiltonian(hamiltonian, config.temperature)
