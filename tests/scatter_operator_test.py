from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import pytest
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator

from coherent_rates.scattering_operator import (
    SparseScatteringOperator,
    as_operator_from_sparse_scattering_operator,
    as_sparse_scattering_operator_from_operator,
    get_periodic_x_operator_sparse,
)
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystem,
    PeriodicSystemConfig,
    get_hamiltonian,
)

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

    _B0 = TypeVar("_B0", bound=BlochBasis[Any])


def get_periodic_x_operator_as_sparse(
    basis: _B0,
    direction: tuple[int, ...] | None,
) -> SparseScatteringOperator[_B0, _B0]:
    direction = tuple(1 for _ in range(basis.ndim)) if direction is None else direction
    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis, direction),
        TupleBasis(basis, basis),
    )
    # Basis of the bloch wavefunction list
    return as_sparse_scattering_operator_from_operator(converted, direction)


@pytest.fixture()
def system() -> PeriodicSystem:
    """Fixture to generate a random n between 2 and 20."""
    return HYDROGEN_NICKEL_SYSTEM_1D


@pytest.fixture()
def config() -> PeriodicSystemConfig:
    """Fixture to generate a random n between 2 and 20."""
    rng = np.random.default_rng()
    shape = rng.integers(1, 10)  # type: ignore unknown
    resolution = rng.integers(3, 10)  # type: ignore unknown
    return PeriodicSystemConfig(
        (shape,),
        (resolution,),
        np.prod(resolution).item(),
        temperature=155,
    )


@pytest.fixture()
def direction() -> tuple[int, ...]:
    """Fixture to generate a random n between 2 and 20."""
    rng = np.random.default_rng()
    return (rng.integers(1, 10),)  # type: ignore unknown


def test_sparse_periodic_x_has_correct_nonzero(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    direction = (1,)
    basis = get_hamiltonian(system, config)["basis"][0]

    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis, direction),
        TupleBasis(basis, basis),
    )

    sparse = get_periodic_x_operator_sparse(basis, direction)

    np.testing.assert_equal(
        np.count_nonzero(np.logical_not(np.isclose(sparse["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(converted["data"], 0))),
    )


def test_sparse_periodic_x_equals_converted(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    direction: tuple[int, ...],
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]
    # Basis of the bloch wavefunction list
    sparse = get_periodic_x_operator_sparse(basis, direction)
    full_as_sparse = get_periodic_x_operator_as_sparse(basis, direction)
    np.testing.assert_array_almost_equal(
        sparse["data"],
        full_as_sparse["data"],
    )


def test_sparse_periodic_x_is_correct_in_momentum_basis(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    direction: tuple[int, ...],
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]

    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)
    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis_k, direction),
        TupleBasis(basis_k, basis_k),
    )
    # Basis of the bloch wavefunction list
    sparse = get_periodic_x_operator_sparse(basis, direction)
    full = convert_operator_to_basis(
        as_operator_from_sparse_scattering_operator(sparse),
        TupleBasis(basis_k, basis_k),
    )
    np.testing.assert_equal(
        np.count_nonzero(np.logical_not(np.isclose(full["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(converted["data"], 0))),
    )

    tolerance = 1e-15
    full["data"][np.abs(full["data"]) < tolerance] = 0
    converted["data"][np.abs(converted["data"]) < tolerance] = 0

    full_data = full["data"].reshape(full["basis"].shape)
    converted_data = converted["data"].reshape(full["basis"].shape)
    np.testing.assert_array_almost_equal(
        full_data,
        converted_data,
    )
