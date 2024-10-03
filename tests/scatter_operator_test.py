from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
import pytest
from surface_potential_analysis.basis.stacked_basis import TupleBasis
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operator import apply_operator_to_state
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.scattering_operator import (
    SparseScatteringOperator,
    apply_scattering_operator_to_state,
    as_operator_from_sparse_scattering_operator,
    as_sparse_scattering_operator_from_operator,
    get_periodic_x_operator_sparse,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.state import get_random_boltzmann_state
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    System,
)

if TYPE_CHECKING:
    from surface_potential_analysis.wavepacket.get_eigenstate import BlochBasis

    _B0 = TypeVar("_B0", bound=BlochBasis[Any])


@pytest.fixture()
def ndim() -> Literal[1, 2]:
    rng = np.random.default_rng()
    return rng.choice([1, 2])


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
    direction = tuple(rng.integers(1, 10) for _ in range(ndim))  # type: ignore unknown
    if ndim == 1:
        shape = rng.integers(1, 10)  # type: ignore unknown
        resolution = rng.integers(3, 10)  # type: ignore unknown
        return PeriodicSystemConfig(
            (shape,),
            (resolution,),
            np.prod(resolution).item(),
            temperature=155,
            direction=direction,
        )
    shape = rng.integers(1, 5, 2)  # type: ignore unknown
    resolution = rng.integers(3, 5, 2)  # type: ignore unknown
    return PeriodicSystemConfig(
        (shape[0], shape[1]),
        (resolution[0], resolution[1]),
        np.prod(resolution).item(),
        temperature=155,
        direction=direction,
    )


def test_sparse_periodic_x_has_correct_nonzero(
    system: System,
    config: PeriodicSystemConfig,
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]

    converted = convert_operator_to_basis(
        get_periodic_x_operator(basis, config.direction),
        TupleBasis(basis, basis),
    )

    sparse = get_periodic_x_operator_sparse(basis, config.direction)

    np.testing.assert_equal(
        np.count_nonzero(np.logical_not(np.isclose(sparse["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(converted["data"], 0))),
    )


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


def test_sparse_periodic_x_equals_converted_full(
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


def test_converted_full_sparse_periodic_x_is_correct_in_momentum_basis(
    system: System,
    config: PeriodicSystemConfig,
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]

    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)
    full = convert_operator_to_basis(
        get_periodic_x_operator(basis_k, config.direction),
        TupleBasis(basis_k, basis_k),
    )
    # Basis of the bloch wavefunction list
    sparse = get_periodic_x_operator_sparse(basis, config.direction)
    sparse_as_full = convert_operator_to_basis(
        as_operator_from_sparse_scattering_operator(sparse),
        TupleBasis(basis_k, basis_k),
    )
    np.testing.assert_equal(
        np.count_nonzero(np.logical_not(np.isclose(full["data"], 0))),
        np.count_nonzero(np.logical_not(np.isclose(sparse_as_full["data"], 0))),
    )

    tolerance = 1e-15
    sparse_as_full["data"][np.abs(sparse_as_full["data"]) < tolerance] = 0
    full["data"][np.abs(full["data"]) < tolerance] = 0

    sparse_as_full_data = sparse_as_full["data"].reshape(sparse_as_full["basis"].shape)
    full_data = full["data"].reshape(sparse_as_full["basis"].shape)
    np.testing.assert_allclose(
        sparse_as_full_data,
        full_data,
        atol=1e-10,
    )


def test_apply_sparse_periodic_x_is_correct(
    system: System,
    config: PeriodicSystemConfig,
) -> None:
    basis = get_hamiltonian(system, config)["basis"][0]

    state = get_random_boltzmann_state(system, config)

    full = get_periodic_x_operator(basis, config.direction)
    sparse = get_periodic_x_operator_sparse(basis, config.direction)

    basis_k = stacked_basis_as_fundamental_momentum_basis(basis)

    np.testing.assert_allclose(
        convert_state_vector_to_basis(apply_operator_to_state(full, state), basis_k)[
            "data"
        ],
        convert_state_vector_to_basis(
            apply_scattering_operator_to_state(sparse, state),
            basis_k,
        )["data"],
        atol=1e-10,
    )
