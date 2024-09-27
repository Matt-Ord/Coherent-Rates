from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Self, TypeVar, cast

import numpy as np
from scipy.constants import (  # type: ignore bad types
    Avogadro,
    electron_volt,
)
from surface_potential_analysis.basis.basis import (
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis,
)
from surface_potential_analysis.basis.basis_like import BasisWithLengthLike
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.util import (
    get_displacements_nx,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_basis,
)
from surface_potential_analysis.potential.potential import Potential
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)

if TYPE_CHECKING:
    from surface_potential_analysis.potential.potential import Potential

_L0Inv = TypeVar("_L0Inv", bound=int)


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
class System(ABC):
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

    @abstractmethod
    def get_potential(
        self: Self,
        shape: tuple[int, ...],
        resolution: tuple[int, ...],
    ) -> Potential[StackedBasisWithVolumeLike[Any, Any, Any]]:
        ...

    def get_potential_basis(
        self: Self,
        shape: tuple[int, ...],
        resolution: tuple[int, ...],
    ) -> StackedBasisWithVolumeLike[Any, Any, Any]:
        return self.get_potential(shape, resolution)["basis"]


class PeriodicSystem(System):
    """A periodic system defined by it's Fundamental Potential."""

    @abstractmethod
    def get_repeating_potential(
        self: Self,
        resolution: tuple[int, ...],
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
        interpolated = self.get_repeating_potential(resolution)
        return _get_extrapolated_potential(interpolated, shape)


class FundamentalPeriodicSystem(PeriodicSystem):
    """A periodic system defined by it's Fundamental Potential."""

    @abstractmethod
    def _get_fundamental_potential(
        self: Self,
    ) -> Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ]:
        ...

    def get_repeating_potential(
        self: Self,
        resolution: tuple[int, ...],
    ) -> Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ]:
        potential = self._get_fundamental_potential()
        interpolated_basis = TupleBasis(
            *tuple(
                TransformedPositionBasis[Any, Any, Any](
                    old.delta_x,
                    old.n,
                    r,
                )
                for (old, r) in zip(
                    cast(
                        Iterator[BasisWithLengthLike[Any, Any, Any]],
                        potential["basis"],
                    ),
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


class FreeSystem(System):
    """A free periodic system."""

    def __init__(self, other: System) -> None:  # noqa: ANN101
        self._other = other
        super().__init__(other.id, 0, other.lattice_constant, other.mass)

    def get_potential(
        self: Self,
        shape: tuple[int, ...],
        resolution: tuple[int, ...],
    ) -> Potential[StackedBasisWithVolumeLike[Any, Any, Any]]:
        other_potential = self._other.get_potential(shape, resolution)
        other_potential["data"] = np.zeros_like(other_potential["data"])
        return other_potential


class PeriodicSystem1d(FundamentalPeriodicSystem):
    """Represents the properties of a 1D Periodic System."""

    def _get_fundamental_potential(
        self: Self,
    ) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
        """Generate potential for a periodic 1D system."""
        delta_x = self.lattice_constant
        axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
        vector = 0.25 * self.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
        return {"basis": TupleBasis(axis), "data": vector}


@dataclass
class PeriodicSystem1dDeep(FundamentalPeriodicSystem):
    """Represents the properties of a 1D Periodic System in a sharp well.

    This produces the sharpest possible well using n_points frequencies
    """

    n_points: int = 5

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        h.update(str(self.barrier_energy).encode())
        h.update(str(self.lattice_constant).encode())
        h.update(str(self.mass).encode())
        h.update(str(self.n_points).encode())

        return int.from_bytes(h.digest(), "big")

    def _get_fundamental_potential(
        self: Self,
    ) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[int]]]:
        n_frequencies = 2 * (self.n_points) + 1

        delta_x = self.lattice_constant
        axis = FundamentalTransformedPositionBasis1d[int](
            np.array([delta_x]),
            n_frequencies,
        )
        vector = np.zeros((n_frequencies,), dtype=np.complex128)
        for i in range(self.n_points):
            vector[i + 1] = 1
            vector[-(i + 1)] = 1
        vector *= -(0.25 * self.barrier_energy * np.sqrt(n_frequencies)) / self.n_points
        vector[0] = 0.5 * self.barrier_energy * np.sqrt(n_frequencies)

        # Fudge factor to get the correct barrier heights
        # the square well converges onto barrier_height / 2, but for n = 1
        # the height is barrier_height
        vector *= (2 * self.n_points) / (1 + self.n_points)

        return {"basis": TupleBasis(axis), "data": vector}


@dataclass
class PeriodicSystem1dDoubleGaussian(PeriodicSystem):
    """Represents the properties of a 1D Periodic System in a sharp well.

    This produces the sharpest possible well using n_points frequencies
    """

    ratio: float = 0.6
    sigma_bottom: int = 16
    sigma_top: int = 4

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        h.update(str(self.barrier_energy).encode())
        h.update(str(self.lattice_constant).encode())
        h.update(str(self.mass).encode())
        h.update(str(self.ratio).encode())
        h.update(str(self.sigma_top).encode())
        h.update(str(self.sigma_bottom).encode())

        return int.from_bytes(h.digest(), "big")

    def get_repeating_potential(
        self: Self,
        resolution: tuple[int, ...],
    ) -> Potential[
        TupleBasisWithLengthLike[
            *tuple[FundamentalTransformedPositionBasis[Any, Any], ...]
        ]
    ]:
        basis = TupleBasis(
            FundamentalTransformedPositionBasis[Any, Any](
                np.array([self.lattice_constant]),
                resolution[0],
            ),
        )
        data = np.zeros(basis.fundamental_n, dtype=np.complex128)
        displacements = get_displacements_nx(basis[0])
        data -= (self.ratio * self.barrier_energy) * np.exp(
            -(displacements[0] ** 2) / (2 * (basis.n / self.sigma_bottom) ** 2),
        )
        data += ((1 - self.ratio) * self.barrier_energy) * np.exp(
            -(displacements[basis.n // 2] ** 2) / (2 * (basis.n / self.sigma_top) ** 2),
        )
        data += self.ratio * self.barrier_energy
        return convert_potential_to_basis(
            {"basis": stacked_basis_as_fundamental_position_basis(basis), "data": data},
            basis,
        )


@dataclass
class PeriodicSystem1dHalfRate(FundamentalPeriodicSystem):
    """Represents the properties of a 1D System with a half-rate fourier component."""

    half_rate_energy: float

    def __hash__(self: Self) -> int:
        h = hashlib.sha256(usedforsecurity=False)
        h.update(self.id.encode())
        h.update(str(self.barrier_energy).encode())
        h.update(str(self.lattice_constant).encode())
        h.update(str(self.mass).encode())
        h.update(str(self.half_rate_energy).encode())

        return int.from_bytes(h.digest(), "big")

    def _get_fundamental_potential(
        self: Self,
    ) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[5]]]]:
        delta_x = self.lattice_constant
        axis = FundamentalTransformedPositionBasis1d[Literal[5]](
            np.array([delta_x]),
            5,
        )

        vector = 0.25 * self.barrier_energy * np.array([2, 0, 1, 1, 0]) * np.sqrt(5)
        vector[[1, -1]] = 0.25 * self.barrier_energy * np.sqrt(5)

        return {"basis": TupleBasis(axis), "data": vector}


class PeriodicSystem2d(FundamentalPeriodicSystem):
    """Represents the properties of a 2D Periodic System."""

    def _get_fundamental_potential(
        self: Self,
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
        data = [[3, 1, 1], [1, 1, 0], [1, 0, 1]]
        vector = self.barrier_energy * np.array(data) / np.sqrt(9)
        return {
            "basis": TupleBasis(
                FundamentalTransformedPositionBasis[Literal[3], Literal[2]](
                    self.lattice_constant * np.array([0, 1]),
                    3,
                ),
                FundamentalTransformedPositionBasis[Literal[3], Literal[2]](
                    self.lattice_constant
                    * np.array(
                        [np.sin(np.pi / 3), np.cos(np.pi / 3)],
                    ),
                    3,
                ),
            ),
            "data": vector.ravel(),
        }


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
    lattice_constant=SODIUM_COPPER_SYSTEM_1D.lattice_constant,
    mass=3.8175458e-26,
)
SODIUM_COPPER_BRIDGE_SYSTEM_DEEP_1D = PeriodicSystem1dDeep(
    id="NaCuDeep",
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=SODIUM_COPPER_BRIDGE_SYSTEM_1D.lattice_constant,
    mass=3.8175458e-26,
)
SODIUM_COPPER_BRIDGE_SYSTEM_GAUSSIAN_1D = PeriodicSystem1dDoubleGaussian(
    id="NaCuGauss",
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=SODIUM_COPPER_SYSTEM_1D.lattice_constant,
    mass=3.8175458e-26,
)
SODIUM_COPPER_BRIDGE_SYSTEM_HALF_RATE_1D = PeriodicSystem1dHalfRate(
    id="NaCuHalfRate",
    barrier_energy=SODIUM_COPPER_BRIDGE_ENERGY,
    lattice_constant=2 * SODIUM_COPPER_SYSTEM_1D.lattice_constant,
    mass=3.8175458e-26,
    half_rate_energy=0.01 * SODIUM_COPPER_BRIDGE_ENERGY,
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
