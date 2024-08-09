from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

import numpy as np
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
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
    """Represents a scattering operator in the bloch basis.

    The
    """

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


def as_operator(
    operator: SparseScatteringOperator[_B0, _B0],
) -> SingleBasisOperator[_B0]:
    # Basis of the bloch wavefunction list
    basis = operator["basis"][0].vectors["basis"][0]
    stacked = operator["data"].reshape(
        basis[0].n,
        basis[0].n,
        -1,
    )

    rolled = np.einsum("ijk,kl->ikjl", stacked, np.eye(stacked.shape[-1])).reshape(
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


def as_scattering_operator(
    operator: SingleBasisOperator[_B0],
    direction: tuple[int, ...],
) -> SparseScatteringOperator[_B0, _B0]:
    # Basis of the bloch wavefunction list
    basis = operator["basis"][0].vectors["basis"][0]
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

    data = np.einsum("ijkj->ikj", rolled).ravel()
    return {"basis": operator["basis"], "direction": direction, "data": data}


def apply_scattering_operator_to_state(
    operator: SparseScatteringOperator[_B0, _B0],
    state: StateVector[_B2],
) -> StateVector[_B0]:
    converted = convert_state_vector_to_basis(state, operator["basis"][1])
    data = np.einsum(
        "ijk,jk->ik",
        # band (out), band (in), bloch k
        operator["data"].reshape(
            operator["basis"][0].vectors["basis"][0][0].n,
            operator["basis"][1].vectors["basis"][0][0].n,
            operator["basis"][1].vectors["basis"][0][1].n,
        ),
        # band (in), bloch k
        converted["data"].reshape(
            operator["basis"][1].vectors["basis"][0].shape,
        ),
    )
    stacked = data.reshape(
        operator["basis"][0].vectors["basis"][0][0].n,
        *operator["basis"][0].vectors["basis"][0][1].shape,
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
    data = np.einsum(
        "ijk,ljk->lik",
        # band (out), band (in), bloch k
        operator["data"].reshape(
            operator["basis"][0].vectors["basis"][0][0].n,
            operator["basis"][1].vectors["basis"][0][0].n,
            operator["basis"][1].vectors["basis"][0][1].n,
        ),
        # list, band, bloch k
        converted["data"].reshape(
            converted["basis"].shape[0],
            *operator["basis"][1].vectors["basis"][0].shape,
        ),
    )

    stacked = data.reshape(
        converted["basis"].shape[0],
        operator["basis"][0].vectors["basis"][0][0].n,
        *operator["basis"][0].vectors["basis"][0][1].shape,
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
    direction: tuple[int, ...] | None,
) -> SparseScatteringOperator[_B0_co, _B0_co]:
    direction = tuple(1 for _ in range(basis.ndim)) if direction is None else direction
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    bloch_wavefunction_basis = basis.wavefunctions["basis"][1]
    shape = bloch_phase_basis.shape
    # Direction of scattering in
    bloch_phase_direction = tuple(
        d % s for (d, s) in zip(direction, shape, strict=True)
    )
    # Direction in the wavefunction space
    bloch_wavefunction_direction = tuple(
        (d // s) for (d, s) in zip(direction, shape, strict=True)
    )

    # Get the periodic x in the basis of bloch wavefunction u(x)
    periodic_x = convert_operator_to_basis(
        get_periodic_x_operator(
            bloch_wavefunction_basis,
            bloch_wavefunction_direction,
        ),
        TupleBasis(bloch_wavefunction_basis, bloch_wavefunction_basis),
    )
    # band (out), band (in), bloch k
    out = np.zeros(
        (band_basis.n, band_basis.n, bloch_phase_basis.n),
        dtype=np.complex128,
    )
    for i, idx_in in enumerate(zip(*BasisUtil(bloch_phase_basis).stacked_nk_points)):
        # Convert periodic x to band basis
        idx_out = tuple(i + s for (i, s) in zip(idx_in, bloch_phase_direction))
        basis_out = basis.basis_at_bloch_k(idx_out)
        basis_in = basis.basis_at_bloch_k(idx_in)
        converted = convert_operator_to_basis(
            periodic_x,
            TupleBasis(basis_out, basis_in),
        )
        out[:, :, i] = converted["data"].reshape(band_basis.n, band_basis.n)
    return {
        "basis": TupleBasis(basis, basis),
        "data": out.ravel(),
        "direction": direction,
    }


@timed
def get_periodic_x_operator_as_sparse(
    basis: _B0_co,
    direction: tuple[int, ...] | None,
) -> SparseScatteringOperator[_B0_co, _B0_co]:
    direction = tuple(1 for _ in range(basis.ndim)) if direction is None else direction
    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis, direction),
        TupleBasis(basis, basis),
    )
    # Basis of the bloch wavefunction list
    return as_scattering_operator(converted, direction)


@timed
def test_get_periodic_x_operator_sparse(
    basis: _B0_co,
    direction: tuple[int, ...] | None,
) -> None:
    direction = tuple(1 for _ in range(basis.ndim)) if direction is None else direction
    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)
    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis, direction),
        TupleBasis(basis, basis),
    )
    # Basis of the bloch wavefunction list
    converted_sparse = get_periodic_x_operator_as_sparse(basis, direction)
    sparse = get_periodic_x_operator_sparse(basis, direction)
    np.testing.assert_equal(
        np.count_nonzero(np.logical_not(np.isclose(converted_sparse["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(converted["data"], 0))),
    )

    operator_k = convert_operator_to_basis(converted, TupleBasis(basis_k, basis_k))
    np.testing.assert_equal(
        operator_k["data"].reshape(operator_k["basis"].shape),
        as_operator(sparse)["data"],
    )
    band_basis = basis.wavefunctions["basis"][0][0]
    bloch_phase_basis = basis.wavefunctions["basis"][0][1]
    shape = (band_basis.n, band_basis.n, bloch_phase_basis.n)
    for phase in range(bloch_phase_basis.n):
        np.testing.assert_array_almost_equal(
            converted_sparse["data"].reshape(shape)[:, :, phase],
            sparse["data"].reshape(shape)[:, :, phase],
        )
