from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pytest
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.scattering_operator import (
    SparseScatteringOperator,
    as_operator_from_sparse_scattering_operator,
    as_sparse_scattering_operator_from_operator,
    get_periodic_x_operator_sparse,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    System,
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
def ndim() -> Literal[1, 2]:
    np.random.default_rng()
    return 1  # rng.choice([1, 2])


@pytest.fixture()
def system(ndim: Literal[1, 2]) -> System:
    """Fixture to generate a random n between 2 and 20."""
    if ndim == 1:
        return HYDROGEN_NICKEL_SYSTEM_1D
    return SODIUM_COPPER_SYSTEM_2D


@pytest.fixture()
def config(ndim: Literal[1, 2]) -> PeriodicSystemConfig:
    """Fixture to generate a random n between 2 and 20."""
    rng = np.random.default_rng()
    if ndim == 1:
        shape = rng.integers(1, 10)  # type: ignore unknown
        resolution = rng.integers(3, 10)  # type: ignore unknown
        return PeriodicSystemConfig(
            (shape,),
            (resolution,),
            np.prod(resolution).item(),
            temperature=155,
            direction=tuple(rng.integers(1, 10) for _ in range(ndim)),  # type: ignore unknown
        )
    shape = rng.integers(1, 5, 2)  # type: ignore unknown
    resolution = rng.integers(3, 5, 2)  # type: ignore unknown
    return PeriodicSystemConfig(
        (shape[0], shape[1]),
        (resolution[0], resolution[1]),
        np.prod(resolution).item(),
        temperature=155,
        direction=tuple(rng.integers(1, 10) for _ in range(ndim)),  # type: ignore unknown
    )


def test_sparse_periodic_x_has_correct_nonzero(
    system: System,
    config: PeriodicSystemConfig,
) -> None:
    direction = (1,)
    basis = get_hamiltonian(system, config)["basis"][0]

    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis, direction),
        TupleBasis(basis, basis),
    )

    sparse = get_periodic_x_operator_sparse(basis, direction)

    np.testing.assert_allclose(
        np.count_nonzero(np.logical_not(np.isclose(sparse["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(converted["data"], 0))),
        atol=1e-10,
    )


def test_sparse_periodic_x_equals_converted(
    system: System,
    config: PeriodicSystemConfig,
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]
    # Basis of the bloch wavefunction list
    sparse = get_periodic_x_operator_sparse(basis, config.direction)
    full_as_sparse = get_periodic_x_operator_as_sparse(basis, config.direction)
    np.testing.assert_array_almost_equal(
        sparse["data"],
        full_as_sparse["data"],
    )


def test_sparse_periodic_x_is_correct_in_momentum_basis(
    system: System,
    config: PeriodicSystemConfig,
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]

    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)
    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis_k, config.direction),
        TupleBasis(basis_k, basis_k),
    )
    # Basis of the bloch wavefunction list
    sparse = get_periodic_x_operator_sparse(basis, config.direction)
    full = convert_operator_to_basis(
        as_operator_from_sparse_scattering_operator(sparse),
        TupleBasis(basis_k, basis_k),
    )
    np.testing.assert_equal(
        np.count_nonzero(np.logical_not(np.isclose(converted["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(full["data"], 0))),
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
