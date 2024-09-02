from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator
from surface_potential_analysis.util.decorators import timed
from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

if TYPE_CHECKING:
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import TupleBasisLike
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    _B0 = TypeVar("_B0", bound=BlochBasis[Any])

    _B1 = TypeVar("_B1", bound=BasisLike[Any, Any])
    _B2 = TypeVar("_B2", bound=BasisLike[Any, Any])

_B0_co = TypeVar("_B0_co", bound=BlochBasis[Any], covariant=True)
_B1_co = TypeVar("_B1_co", bound=BlochBasis[Any], covariant=True)


class SparseScatteringOperator(TypedDict, Generic[_B0_co, _B1_co]):
    """Represents an operator in the (sparse) scattering operator basis."""

    basis: TupleBasisLike[_B0_co, _B1_co]
    """The original basis of the operator."""

    data: np.ndarray[tuple[int], np.dtype[np.complex128]]
    """The operator stored in sparse form.

    The data is stored in band (out), band (in), bloch k such that the full
    original data can be found by doing

    ```python
    full = np.einsum("ijk->ikjk", operator["data"].reshape(
            operator["basis"][0].vectors["basis"][0].n,
            operator["basis"][1].vectors["basis"][0].n,
            operator["basis"][1].vectors["basis"][1].n,
        ),)
    stacked = full.reshape(
        operator["basis"][0].vectors["basis"][0].n,
        *operator["basis"][0].vectors["basis"][1].shape,
        operator["basis"][1].vectors["basis"][0].n,
        *operator["basis"][1].vectors["basis"][1].shape,
    )
    data = np.roll(
        stacked,
        direction,
        axis=tuple(1 + i for i in range(basis[1].ndim)),
    ).ravel()
    ```
    """

    direction: tuple[int, ...]
    """The direction of scattering"""


def as_operator_from_sparse_scattering_operator(
    operator: SparseScatteringOperator[_B0, _B0],
) -> SingleBasisOperator[_B0]:
    # Basis of the bloch wavefunction list
    basis = operator["basis"][0].wavefunctions["basis"][0]
    stacked = operator["data"].reshape(
        basis[0].n,
        basis[0].n,
        -1,
    )

    rolled = np.einsum("ijk,kl->ikjl", stacked, np.eye(stacked.shape[-1])).reshape(  # type: ignore bad types
        basis[0].n,
        *basis[1].shape,
        basis[0].n,
        *basis[1].shape,
    )
    data = np.roll(
        rolled,
        operator["direction"],
        axis=tuple(1 + i for i in range(basis[1].ndim)),
    ).ravel()

    return {"basis": operator["basis"], "data": data}


def as_sparse_scattering_operator_from_operator(
    operator: SingleBasisOperator[_B0],
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0, _B0]:
    # Basis of the bloch wavefunction list
    basis = operator["basis"][0].wavefunctions["basis"][0]
    stacked = operator["data"].reshape(
        basis[0].n,
        *basis[1].shape,
        basis[0].n,
        *basis[1].shape,
    )

    rolled = np.roll(
        stacked,
        tuple(-d for d in direction),
        axis=tuple(1 + i for i in range(basis[1].ndim)),
    ).reshape(*basis.shape, *basis.shape)

    data = np.einsum("ijkj->ikj", rolled).ravel()  # type: ignore bad types
    return {"basis": operator["basis"], "direction": direction, "data": data}


def apply_scattering_operator_to_state(
    operator: SparseScatteringOperator[_B0, _B0],
    state: StateVector[_B2],
) -> StateVector[_B0]:
    converted = convert_state_vector_to_basis(state, operator["basis"][1])
    data = np.einsum(  # type: ignore bad types
        "ijk,jk->ik",
        # band (out), band (in), bloch k
        operator["data"].reshape(
            operator["basis"][0].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][1].n,
        ),
        # band (in), bloch k
        converted["data"].reshape(
            operator["basis"][1].wavefunctions["basis"][0].shape,
        ),
    )
    stacked = data.reshape(
        operator["basis"][0].wavefunctions["basis"][0][0].n,
        *operator["basis"][0].wavefunctions["basis"][0][1].shape,
    )
    rolled = np.roll(
        stacked,
        operator["direction"],
        axis=tuple(range(1, stacked.ndim)),
    ).ravel()

    return {"basis": operator["basis"][0], "data": rolled}


def apply_scattering_operator_to_states(
    operator: SparseScatteringOperator[_B0, _B0],
    states: StateVectorList[_B2, _B1],
) -> StateVectorList[_B2, _B0]:
    converted = convert_state_vector_list_to_basis(states, operator["basis"][1])
    data = np.einsum(  # type: ignore bad types
        "ijk,ljk->lik",
        # band (out), band (in), bloch k
        operator["data"].reshape(
            operator["basis"][0].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][0].n,
            operator["basis"][1].wavefunctions["basis"][0][1].n,
        ),
        # list, band, bloch k
        converted["data"].reshape(
            converted["basis"].shape[0],
            *operator["basis"][1].wavefunctions["basis"][0].shape,
        ),
    )

    stacked = data.reshape(
        converted["basis"].shape[0],
        operator["basis"][0].wavefunctions["basis"][0][0].n,
        *operator["basis"][0].wavefunctions["basis"][0][1].shape,
    )
    rolled = np.roll(
        stacked,
        operator["direction"],
        axis=tuple(range(2, stacked.ndim)),
    ).ravel()

    return {
        "basis": TupleBasis(states["basis"][0], operator["basis"][0]),
        "data": rolled,
    }


@timed
def get_periodic_x_operator_sparse(
    basis: _B0_co,
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0_co, _B0_co]:
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    # band (out), band (in), bloch k
    out = np.zeros(
        (band_basis.n, band_basis.n, bloch_phase_basis.n),
        dtype=np.complex128,
    )
    stacked_nk_points = BasisUtil(bloch_phase_basis).stacked_nk_points
    for i, nk_in in enumerate(zip(*stacked_nk_points)):
        # Find the bloch k of the scattered state
        util = BasisUtil(basis.wavefunctions["basis"][0][1])
        idx_out = util.get_flat_index(
            tuple(j + s for (j, s) in zip(nk_in, direction)),
            mode="wrap",
        )
        nk_out = tuple(
            k[i] for k, i in zip(stacked_nk_points, util.get_stacked_index(idx_out))
        )

        # Direction in the wavefunction space
        # Not this is not the same for all idx_in, idx_out
        bloch_wavefunction_basis = basis.wavefunctions["basis"][1]
        bloch_wavefunction_direction = tuple(
            (d - (out - in_)) // s
            for (d, in_, out, s) in zip(
                direction,
                nk_in,
                nk_out,
                bloch_phase_basis.shape,
                strict=True,
            )
        )

        # Get the periodic x in the basis of bloch wavefunction u(x)
        basis_in = basis.basis_at_bloch_k(nk_in)
        basis_out = basis.basis_at_bloch_k(nk_out)
        periodic_x = convert_operator_to_basis(
            get_periodic_x_operator(
                bloch_wavefunction_basis,
                bloch_wavefunction_direction,
            ),
            TupleBasis(basis_out, basis_in),
        )

        out[:, :, i] = periodic_x["data"].reshape(band_basis.n, band_basis.n)
    return {
        "basis": TupleBasis(basis, basis),
        "data": out.ravel(),
        "direction": direction,
    }


@timed
def get_energy_change_operator_sparse(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0, _B0]:
    basis = hamiltonian["basis"][0]
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    # band (out), band (in), bloch k
    out = np.zeros(
        (band_basis.n, band_basis.n, bloch_phase_basis.n),
        dtype=np.complex128,
    )
    stacked_nk_points = BasisUtil(bloch_phase_basis).stacked_nk_points

    eigenvalues = hamiltonian["data"].reshape(band_basis.n, bloch_phase_basis.n)
    for i, nk_in in enumerate(zip(*stacked_nk_points)):
        # Find the bloch k of the scattered state
        util = BasisUtil(basis.wavefunctions["basis"][0][1])
        j = util.get_flat_index(
            tuple(j + s for (j, s) in zip(nk_in, direction)),
            mode="wrap",
        )

        out[:, :, i] = eigenvalues[:, np.newaxis, j] - eigenvalues[np.newaxis, :, i]

    return {
        "basis": hamiltonian["basis"],
        "data": out.ravel(),
        "direction": direction,
    }


@timed
def get_instrument_biased_periodic_x(
    hamiltonian: SingleBasisDiagonalOperator[_B0],
    direction: tuple[int, ...] | None,
    energy_range: tuple[float, float],
) -> SparseScatteringOperator[_B0, _B0]:
    direction = (
        tuple(1 for _ in range(hamiltonian["basis"][0].ndim))
        if direction is None
        else direction
    )

    periodic_x = get_periodic_x_operator_sparse(hamiltonian["basis"][0], direction)

    scattered_energy = get_energy_change_operator_sparse(hamiltonian, direction)
    min_energy, max_energy = energy_range
    mask = np.logical_and(
        min_energy < scattered_energy["data"],
        scattered_energy["data"] < max_energy,
    )
    periodic_x["data"][np.logical_not(mask)] = 0

    return periodic_x
