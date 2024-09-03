from __future__ import annotations

import hashlib
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Literal, Self, TypeVar, cast

import numpy as np
from scipy.constants import (  # type: ignore bad types
    Avogadro,
    Boltzmann,
    electron_volt,
    hbar,
)
from surface_potential_analysis.basis.basis import (
    FundamentalPositionBasis,
    FundamentalTransformedBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis.basis_like import BasisWithLengthLike
from surface_potential_analysis.basis.evenly_spaced_basis import (
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
from surface_potential_analysis.basis.util import BasisUtil
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
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.get_eigenstate import (
    BlochBasis,
    get_full_bloch_hamiltonian,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListWithEigenvaluesList,
    generate_wavepacket,
)

if TYPE_CHECKING:
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
    system: PeriodicSystem1d,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    """Generate potential for a periodic 1D system."""
    delta_x = system.lattice_constant
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def _get_fundamental_potential_2d(
    system: PeriodicSystem2d,
) -> Potential[
    TupleBasis[
        FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
        FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
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
    # We want the simplest possible potential in 2d with symmetry
    # (x0,x1) -> (x1,x0)
    # (x0,x1) -> (-x0,x1)
    # (x0,x1) -> (x0,-x1)
    # We therefore occupy G = +-K0, +-K1, +-(K0+K1) equally
    data = [[0, 1, 1], [1, 1, 0], [1, 0, 1]]
    vector = system.barrier_energy * np.array(data) / np.sqrt(9)
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


@dataclass
class PeriodicSystem:
    """Represents the properties of a Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float

    def with_mass(self: Self, mass: float) -> Self:
        copied = copy(self)
        copied.mass = mass
        return copied

    def with_barrier_energy(self: Self, barrier_energy: float) -> Self:
        copied = copy(self)
        copied.barrier_energy = barrier_energy
        return copied

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        h.update(str(self.barrier_energy).encode())
        h.update(str(self.lattice_constant).encode())
        h.update(str(self.mass).encode())

        return int.from_bytes(h.digest(), "big")

    def get_fundamental_potential(
        self: Self,
    ) -> Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ]:
        ...

    def get_potential(
        self: Self,
        shape: tuple[int, ...],
        resolution: tuple[int, ...],
    ) -> Potential[StackedBasisWithVolumeLike[Any, Any, Any]]:
        potential = self.get_fundamental_potential()
        interpolated = _get_interpolated_potential(potential, resolution)

        return _get_extrapolated_potential(interpolated, shape)

    def get_potential_basis(
        self: Self,
        shape: tuple[int, ...],
        resolution: tuple[int, ...],
    ) -> StackedBasisWithVolumeLike[Any, Any, Any]:
        return self.get_potential(shape, resolution)["basis"]


class FreeSystem(PeriodicSystem):
    """A free periodic system."""

    def __init__(self, other: PeriodicSystem) -> None:  # noqa: ANN101, D107
        self._other = other
        super().__init__(other.id, 0, other.lattice_constant, other.mass)

    def get_fundamental_potential(
        self: Self,
    ) -> Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ]:
        other_potential = self._other.get_fundamental_potential()
        other_potential["data"] = np.zeros_like(other_potential["data"])
        return other_potential


class PeriodicSystem1d(PeriodicSystem):
    """Represents the properties of a 1D Periodic System."""

    def get_fundamental_potential(
        self: Self,
    ) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
        return _get_fundamental_potential_1d(self)


class PeriodicSystem2d(PeriodicSystem):
    """Represents the properties of a 2D Periodic System."""

    def get_fundamental_potential(
        self: Self,
    ) -> Potential[
        TupleBasis[
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
            FundamentalTransformedPositionBasis[Literal[3], Literal[2]],
        ]
    ]:
        return _get_fundamental_potential_2d(self)


_DEFAULT_DIRECTION = ()


@dataclass
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int, ...]
    resolution: tuple[int, ...]
    truncation: int | None = None
    temperature: float = field(default=150, kw_only=True)
    scattered_energy_range: tuple[float, float] = field(
        default=(-np.inf, np.inf),
        kw_only=True,
    )
    direction: tuple[int, ...] = field(default=_DEFAULT_DIRECTION, kw_only=True)

    def __post_init__(self: Self) -> None:
        if self.direction is _DEFAULT_DIRECTION:
            self.direction = tuple(0 for _ in self.shape)

    def with_direction(self: Self, direction: tuple[int, ...]) -> Self:
        copied = copy(self)
        copied.direction = direction
        return copied

    def with_temperature(self: Self, temperature: float) -> Self:
        copied = copy(self)
        copied.temperature = temperature
        return copied

    def with_resolution(self: Self, resolution: tuple[int, ...]) -> Self:
        copied = copy(self)
        copied.resolution = resolution
        return copied

    def with_shape(self: Self, shape: tuple[int, ...]) -> Self:
        copied = copy(self)
        copied.shape = shape
        return copied

    def with_truncation(self: Self, truncation: int | None) -> Self:
        copied = copy(self)
        copied.truncation = truncation
        return copied

    def with_scattered_energy_range(
        self: Self,
        energy_range: tuple[float, float],
    ) -> Self:
        copied = copy(self)
        copied.scattered_energy_range = energy_range
        return copied

    @property
    def n_bands(self: Self) -> int:
        """Total number of bands.

        Parameters
        ----------
        self : Self

        Returns
        -------
        int

        """
        return (
            np.prod(self.resolution).item()
            if self.truncation is None
            else self.truncation
        )

    def __hash__(self: Self) -> int:
        return hash(
            (
                self.shape,
                self.resolution,
                self.n_bands,
                self.temperature,
                self.direction,
                self.scattered_energy_range,
            ),
        )


HYDROGEN_NICKEL_SYSTEM_1D = PeriodicSystem1d(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)

HYDROGEN_NICKEL_SYSTEM_2D = PeriodicSystem2d(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)


# see <https://www.sciencedirect.com/science/article/pii/S0039602897000897>
SODIUM_COPPER_BRIDGE_ENERGY = (416.78 - 414.24) * 1e3 / Avogadro
SODIUM_COPPER_SYSTEM_2D = PeriodicSystem2d(
    id="NaCu",
    barrier_energy=9 * SODIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=2.558e-10,
    mass=3.8175458e-26,
)
SODIUM_COPPER_SYSTEM_1D = PeriodicSystem1d(
    id="NaCu",
    barrier_energy=55e-3 * electron_volt,
    lattice_constant=(1 / np.sqrt(3)) * SODIUM_COPPER_SYSTEM_2D.lattice_constant,
    mass=3.8175458e-26,
)
SODIUM_COPPER_BRIDGE_SYSTEM_1D = PeriodicSystem1d(
    id="NaCuB",
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=(1 / np.sqrt(3)) * SODIUM_COPPER_SYSTEM_2D.lattice_constant,
    mass=3.8175458e-26,
)


# see <https://www.sciencedirect.com/science/article/pii/S0039602897000897>
LITHIUM_COPPER_BRIDGE_ENERGY = (477.16 - 471.41) * 1e3 / Avogadro
LITHIUM_COPPER_SYSTEM_2D = PeriodicSystem2d(
    id="LiCu",
    barrier_energy=9 * LITHIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
)
LITHIUM_COPPER_SYSTEM_1D = PeriodicSystem1d(
    id="LiCu",
    barrier_energy=45e-3 * electron_volt,
    lattice_constant=(1 / np.sqrt(3)) * LITHIUM_COPPER_SYSTEM_2D.lattice_constant,
    mass=1.152414898e-26,
)
LITHIUM_COPPER_BRIDGE_SYSTEM_1D = PeriodicSystem1d(
    id="LiCuB",
    barrier_energy=LITHIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=(1 / np.sqrt(3)) * LITHIUM_COPPER_SYSTEM_2D.lattice_constant,
    mass=1.152414898e-26,
)


def _get_full_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[_L0Inv, ...],
    resolution: tuple[_L0Inv, ...],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[
        *tuple[FundamentalTransformedPositionBasis[int, int], ...]
    ],
]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction
    potential = system.get_potential(shape, resolution)

    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_momentum_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, system.mass, bloch_fraction)


def get_bloch_wavefunctions(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> BlochWavefunctionListWithEigenvaluesList[
    TruncatedBasis[int, int],
    TupleBasisLike[*tuple[FundamentalTransformedBasis[Any], ...]],
    StackedBasisWithVolumeLike[Any, Any, Any],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[StackedBasisWithVolumeLike[Any, Any, Any]]:
        return _get_full_hamiltonian(
            system,
            tuple(1 for _ in config.shape),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    return generate_wavepacket(
        hamiltonian_generator,
        band_basis=TruncatedBasis(config.n_bands, np.prod(config.resolution).item()),
        list_basis=fundamental_transformed_stacked_basis_from_shape(config.shape),
    )


@timed
def get_hamiltonian(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalOperator[BlochBasis[TruncatedBasis[int, int]]]:
    wavefunctions = get_bloch_wavefunctions(system, config)

    return get_full_bloch_hamiltonian(wavefunctions)


_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def solve_schrodinger_equation(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> StateVectorList[
    _AX0Inv,
    BlochBasis[TruncatedBasis[int, int]],
]:
    hamiltonian = get_hamiltonian(system, config)
    return solve_schrodinger_equation_diagonal(initial_state, times, hamiltonian)


def get_step_state_1d(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
    fraction: float | None,
) -> StateVector[TupleBasisWithLengthLike[FundamentalPositionBasis[Any, Literal[1]]]]:
    potential = system.get_potential(config.shape, config.resolution)
    basis = stacked_basis_as_fundamental_position_basis(potential["basis"])

    initial_state: StateVector[Any] = {
        "basis": basis,
        "data": np.zeros(basis.n, dtype=np.complex128),
    }
    n_non_zero = 1 if fraction is None else int(fraction * basis.n)
    for i in range(n_non_zero):
        initial_state["data"][i] = 1 / np.sqrt(n_non_zero)
    return initial_state


_SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


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
