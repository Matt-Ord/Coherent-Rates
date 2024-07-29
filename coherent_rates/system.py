from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Self, TypeVar, cast

import numpy as np
from scipy.constants import Boltzmann, electron_volt, hbar
from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis,
)
from surface_potential_analysis.basis.basis_like import BasisWithLengthLike
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
from surface_potential_analysis.basis.util import (
    get_displacements_x,
    get_twice_average_nx,
)
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
    fundamental_transformed_stacked_basis_from_shape,
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


def _get_fundamental_potential_1d(
    system: PeriodicSystem1D,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    """Generate potential for a periodic 1D system."""
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def _get_fundamental_potential_2d(
    system: PeriodicSystem2D,
) -> Potential[
    TupleBasis[
        FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
        FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
    ]
]:
    # We want the simplest possible potential in 2d with symmetry
    # (x0,x1) -> (x1,x0)
    # (x0,x1) -> (-x0,x1)
    # (x0,x1) -> (x0,-x1)
    # We therefore occupy G = +-K0, +-K1, +-(K0+K1) equally
    data = [[0, 1, 1], [1, 1, 0], [1, 0, 1]]
    vector = 0.5 * system.barrier_energy * np.array(data) / np.sqrt(9)
    return {
        "basis": TupleBasis(
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]](
                system.lattice_constant * np.array([0, 1]),
                3,
            ),
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]](
                system.lattice_constant
                * np.array(
                    [np.sin(np.pi / 3), np.cos(np.pi / 3)],
                ),
                3,
            ),
        ),
        "data": vector.ravel(),
    }


@dataclass
class PeriodicSystem:
    """Represents the properties of a Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float

    def __hash__(self: Self) -> int:  # noqa: D105
        return hash((self.id, self.barrier_energy, self.lattice_constant, self.mass))

    def get_fundamental_potential(
        self: Self,
    ) -> Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ]:
        ...

    def potential(
        self: Self,
        shape: tuple[int, ...],
        resolution: tuple[int, ...],
    ) -> Potential[StackedBasisWithVolumeLike[Any, Any, Any]]:
        potential = self.get_fundamental_potential()
        interpolated = _get_interpolated_potential(potential, resolution)

        return _get_extrapolated_potential(interpolated, shape)

    def as_free_system(self: Self) -> PeriodicSystem:
        ...


@dataclass
class PeriodicSystem1D(PeriodicSystem):
    """Represents the properties of a 1D Periodic System."""

    def get_fundamental_potential(
        self: Self,
    ) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
        return _get_fundamental_potential_1d(self)

    def as_free_system(self: Self) -> PeriodicSystem1D:
        return PeriodicSystem1D(self.id, 0, self.lattice_constant, self.mass)


@dataclass
class PeriodicSystem2D(PeriodicSystem):
    def get_fundamental_potential(
        self: Self,
    ) -> Potential[
        TupleBasis[
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
        ]
    ]:
        return _get_fundamental_potential_2d(self)

    def as_free_system(self: Self) -> PeriodicSystem2D:
        return PeriodicSystem2D(self.id, 0, self.lattice_constant, self.mass)


@dataclass
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int, ...]
    resolution: tuple[int, ...]
    n_bands: int
    temperature: float

    def __hash__(self: Self) -> int:  # noqa: D105
        return hash((self.shape, self.resolution, self.n_bands, self.temperature))


HYDROGEN_NICKEL_SYSTEM_1D = PeriodicSystem1D(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)

HYDROGEN_NICKEL_SYSTEM_2D = PeriodicSystem2D(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)

SODIUM_COPPER_SYSTEM_1D = PeriodicSystem1D(
    id="NaCu",
    barrier_energy=55e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=3.8175458e-26,
)

LITHIUM_COPPER_SYSTEM_1D = PeriodicSystem1D(
    id="LiCu",
    barrier_energy=45e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
)


def _get_interpolated_potential(
    potential: Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ],
    resolution: tuple[_L0Inv, ...],
) -> Potential[
    TupleBasisWithLengthLike[*tuple[FundamentalTransformedPositionBasis[Any, Any], ...]]
]:
    interpolated_basis = TupleBasis(
        *tuple(
            TransformedPositionBasis[Any, Any, Any](
                old.delta_x,
                old.n,
                r,
            )
            for (old, r) in zip(
                cast(Iterator[BasisWithLengthLike[Any, Any, Any]], potential["basis"]),
                resolution,
            )
        ),
    )

    scaled_potential = potential["data"] * np.sqrt(
        interpolated_basis.fundamental_n / potential["basis"].n,
    )

    return convert_potential_to_basis(
        {"basis": interpolated_basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(interpolated_basis),
    )


def _get_extrapolated_potential(
    potential: Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ],
    shape: tuple[_L0Inv, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    extrapolated_basis = TupleBasis(
        *tuple(
            EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any](
                old.delta_x * s,
                n=old.n,
                step=s,
                offset=0,
            )
            for (old, s) in zip(
                cast(Iterator[BasisWithLengthLike[Any, Any, Any]], potential["basis"]),
                shape,
            )
        ),
    )

    scaled_potential = potential["data"] * np.sqrt(
        extrapolated_basis.fundamental_n / potential["basis"].n,
    )

    return {"basis": extrapolated_basis, "data": scaled_potential}


def get_potential_1d(
    system: PeriodicSystem1D,
    shape: tuple[int, ...],
    resolution: tuple[int, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    potential = _get_fundamental_potential_1d(system)
    interpolated = _get_interpolated_potential(potential, resolution)

    return _get_extrapolated_potential(interpolated, shape)


def get_potential_2d(
    system: PeriodicSystem2D,
    shape: tuple[_L0Inv, ...],
    resolution: tuple[int, ...],
) -> Potential[
    TupleBasisWithLengthLike[
        *tuple[EvenlySpacedTransformedPositionBasis[Any, Any, Any, Any], ...]
    ]
]:
    """Generate potential for 2D periodic system, for 111 plane of FCC lattice.

    Expression for potential from:
    [1] D. J. Ward
        A study of spin-echo lineshapes in helium atom scattering from adsorbates.
    [2]S. P. Rittmeyer et al
        Energy Dissipation during Diffusion at Metal Surfaces:
        Disentangling the Role of Phonons vs Electron-Hole Pairs.
    """
    potential = _get_fundamental_potential_2d(system)
    interpolated = _get_interpolated_potential(potential, resolution)
    return _get_extrapolated_potential(interpolated, shape)


def _get_full_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[_L0Inv, ...],
    resolution: tuple[_L0Inv, ...],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis[int, int], ...]],
]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction
    potential = system.potential(shape, resolution)

    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_position_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, system.mass, bloch_fraction)


def get_bloch_wavefunctions(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> BlochWavefunctionListWithEigenvaluesList[
    EvenlySpacedBasis[int, int, int],
    TupleBasisLike[*tuple[FundamentalTransformedBasis[Any], ...]],
    StackedBasisWithVolumeLike[Any, Any, Any],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[StackedBasisWithVolumeLike[Any, Any, Any]]:
        return _get_full_hamiltonian(
            system,
            tuple(1 for _ in range(len(config.shape))),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    return generate_wavepacket(
        hamiltonian_generator,
        band_basis=EvenlySpacedBasis(config.n_bands, 1, 0),
        list_basis=fundamental_transformed_stacked_basis_from_shape(config.shape),
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


def get_step_state_1d(
    system: PeriodicSystem1D,
    config: PeriodicSystemConfig,
    fraction: float | None,
) -> StateVector[TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]]:
    potential = get_potential_1d(system, config.shape, config.resolution)
    basis = stacked_basis_as_fundamental_position_basis(potential["basis"])

    initial_state: StateVector[Any] = {
        "basis": basis,
        "data": np.zeros(basis.n, dtype=np.complex128),
    }
    n_non_zero = 1 if fraction is None else int(fraction * basis.n)
    for i in range(n_non_zero):
        initial_state["data"][i] = 1 / np.sqrt(n_non_zero)
    return initial_state


def get_gaussian_state_1d(
    system: PeriodicSystem1D,
    config: PeriodicSystemConfig,
    fraction: float,
) -> StateVector[TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]]:
    potential = get_potential_1d(system, config.shape, config.resolution)
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
    system: PeriodicSystem1D,
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
    ccc

    Returns:
    -------
        SingleBasisOperator[Any]: _description_

    """
    potential = get_potential_1d(system, config.shape, config.resolution)

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
