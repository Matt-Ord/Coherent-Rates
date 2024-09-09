from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from surface_potential_analysis.basis.basis import (
    FundamentalTransformedBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_basis,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_transformed_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.util.decorators import npy_cached_dict, timed
from surface_potential_analysis.wavepacket.get_eigenstate import (
    BlochBasis,
    get_full_bloch_hamiltonian,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListWithEigenvaluesList,
    generate_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis import (
        FundamentalTransformedPositionBasis,
    )
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
        TupleBasisLike,
        TupleBasisWithLengthLike,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    from coherent_rates.config import PeriodicSystemConfig
    from coherent_rates.system import System

_L0Inv = TypeVar("_L0Inv", bound=int)


@timed
def _get_full_hamiltonian(
    system: System,
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


def _get_bloch_wavefunctions_path(
    system: System,
    config: PeriodicSystemConfig,
) -> Path:
    return Path(f"data/{hash((system,config))}.wavefunctions.npz")


@npy_cached_dict(_get_bloch_wavefunctions_path, load_pickle=True)
def get_bloch_wavefunctions(
    system: System,
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
    system: System,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalOperator[BlochBasis[TruncatedBasis[int, int]]]:
    wavefunctions = get_bloch_wavefunctions.call_uncached(system, config)

    return get_full_bloch_hamiltonian(wavefunctions)


_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def solve_schrodinger_equation(
    system: System,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> StateVectorList[
    _AX0Inv,
    BlochBasis[TruncatedBasis[int, int]],
]:
    hamiltonian = get_hamiltonian(system, config)
    return solve_schrodinger_equation_diagonal(initial_state, times, hamiltonian)
