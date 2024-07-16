from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import Boltzmann, electron_volt, hbar
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedBasis,
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisLike,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.time_basis_like import (
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.basis.util import get_displacements_x
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_basis,
    convert_potential_to_position_basis,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.state_vector import calculate_normalization
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_full_bloch_hamiltonian,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListWithEigenvaluesList,
    generate_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.potential.potential import Potential
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

_L0Inv = TypeVar("_L0Inv", bound=int)


@dataclass
class PeriodicSystem:
    """Represents the properties of a 1D Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float


@dataclass
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int]
    resolution: tuple[int]
    n_bands: int
    temperature: float


HYDROGEN_NICKEL_SYSTEM = PeriodicSystem(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)

HYDROGEN_FREE = PeriodicSystem(
    id="HFree",
    barrier_energy=0,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)

SODIUM_COPPER_SYSTEM = PeriodicSystem(
    id="NaCu",
    barrier_energy=55e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=3.8175458e-26,
)

SODIUM_FREE = PeriodicSystem(
    id="NaCu",
    barrier_energy=0,
    lattice_constant=3.615e-10,
    mass=3.8175458e-26,
)

LITHIUM_COPPER_SYSTEM = PeriodicSystem(
    id="LiCu",
    barrier_energy=45e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
)

LITHIUM_FREE = PeriodicSystem(
    id="LiCu",
    barrier_energy=0,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
)


def _get_base_potential(
    system: PeriodicSystem,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def _get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[
    TupleBasisWithLengthLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = _get_base_potential(system)
    old = potential["basis"][0]
    basis = TupleBasis(
        TransformedPositionBasis1d[_L0Inv, Literal[3]](
            old.delta_x,
            old.n,
            resolution[0],
        ),
    )
    scaled_potential = potential["data"] * np.sqrt(resolution[0] / old.n)
    return convert_potential_to_basis(
        {"basis": basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(basis),
    )


def _get_extended_potential(
    system: PeriodicSystem,
    shape: tuple[int],
    resolution: tuple[int],
) -> Potential[
    TupleBasisWithLengthLike[
        EvenlySpacedTransformedPositionBasis[int, int, Literal[0], Literal[1]]
    ]
]:
    interpolated = _get_interpolated_potential(system, resolution)
    old = interpolated["basis"][0]
    basis = TupleBasis(
        EvenlySpacedTransformedPositionBasis[int, int, Literal[0], Literal[1]](
            old.delta_x * shape[0],
            n=old.n,
            step=shape[0],
            offset=0,
        ),
    )
    scaled_potential = interpolated["data"] * np.sqrt(basis.fundamental_n / old.n)

    return {"basis": basis, "data": scaled_potential}


def get_potential(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> Potential[
    TupleBasisWithLengthLike[
        EvenlySpacedTransformedPositionBasis[int, int, Literal[0], Literal[1]]
    ]
]:
    return _get_extended_potential(system, config.shape, config.resolution)


def _get_full_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[int],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[StackedBasisWithVolumeLike[Any, Any, Any]]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction

    potential = _get_extended_potential(system, shape, resolution)
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_momentum_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, system.mass, bloch_fraction)


def get_bloch_wavefunctions(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> BlochWavefunctionListWithEigenvaluesList[
    EvenlySpacedBasis[int, int, int],
    TupleBasisLike[FundamentalBasis[int]],
    StackedBasisWithVolumeLike[Any, Any, Any],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[StackedBasisWithVolumeLike[Any, Any, Any]]:
        return _get_full_hamiltonian(
            system,
            (1,),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    return generate_wavepacket(
        hamiltonian_generator,
        save_bands=EvenlySpacedBasis(config.n_bands, 1, 0),
        list_basis=fundamental_stacked_basis_from_shape(config.shape),
    )


def get_hamiltonian(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalOperator[ExplicitStackedBasisWithLength[Any, Any]]:
    wavefunctions = get_bloch_wavefunctions(system, config)

    return get_full_bloch_hamiltonian(wavefunctions)


_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def solve_schrodinger_equation(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> StateVectorList[_AX0Inv, ExplicitStackedBasisWithLength[Any, Any]]:
    hamiltonian = get_hamiltonian(system, config)
    return solve_schrodinger_equation_diagonal(initial_state, times, hamiltonian)


def get_step_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    fraction: float | None,
) -> StateVector[TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]]:
    potential = get_potential(system, config)
    basis = stacked_basis_as_fundamental_position_basis(potential["basis"])

    initial_state: StateVector[Any] = {
        "basis": basis,
        "data": np.zeros(basis.n, dtype=np.complex128),
    }
    n_non_zero = 1 if fraction is None else int(fraction * basis.n)
    for i in range(n_non_zero):
        initial_state["data"][i] = 1 / np.sqrt(n_non_zero)
    return initial_state


def get_gaussian_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    fraction: float,
) -> StateVector[TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]]:
    potential = get_potential(system, config)
    basis = stacked_basis_as_fundamental_position_basis(potential["basis"])

    size = basis.n
    width = size * fraction
    data = np.zeros(basis.n, dtype=np.complex128)

    for i in range(size):
        data[i] = -np.square(i - size / 2) / (2 * width * width)

    data = np.exp(data)
    initial_state: StateVector[Any] = {
        "basis": basis,
        "data": data,
    }
    initial_state["data"] = initial_state["data"] / np.sqrt(
        calculate_normalization(
            initial_state,
        ),
    )
    return initial_state


def get_cl_operator(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisOperator[Any]:
    """Generate the operator for the stationary Caldeira-Leggett solution.

    Follows the formula from eqn 3.421 in
    https://doi.org/10.1093/acprof:oso/9780199213900.001.0001

    Args:
    ----
        system (PeriodicSystem): _description_
        config (PeriodicSystemConfig): _description_
        temperature (float): _description_

    Returns:
    -------
        SingleBasisOperator[Any]: _description_

    """
    potential = get_potential(system, config)

    basis_x = stacked_basis_as_fundamental_position_basis(potential["basis"])
    converted_potential = convert_potential_to_position_basis(potential)

    displacements = get_displacements_x(basis_x)
    average_nx = get_twice_average_nx(basis_x)

    n_states = basis_x.n

    # if average_nx // 2 is an integer, this is just V[i]
    # Otherwise this averages V[floor average_nx // 2] and V[ceil average_nx // 2]
    floor_idx = np.floor(average_nx[0] / 2).astype(np.int_) % n_states
    ceil_idx = np.ceil(average_nx[0] / 2).astype(np.int_) % n_states
    average_potential = (
        converted_potential["data"][floor_idx] + converted_potential["data"][ceil_idx]
    ) / 2

    # density matrix in position basis (un-normalized)
    # \rho_s(x, x') = N \exp(-V(x+x' / 2) / kt - mkt(x-x')^2/ 2hbar^2)
    matrix = np.exp(
        -(
            (average_potential / (Boltzmann * config.temperature))
            + (
                (system.mass * Boltzmann * config.temperature * displacements**2)
                / (2 * hbar**2)
            )
        ),
    )

    return {
        "basis": TupleBasis(converted_potential["basis"], converted_potential["basis"]),
        "data": matrix.ravel(),
    }
