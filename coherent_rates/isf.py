from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.operator.operations import apply_operator_to_states
from surface_potential_analysis.operator.operator import (
    SingleBasisDiagonalOperator,
    SingleBasisOperator,
    apply_operator_to_state,
)
from surface_potential_analysis.state_vector.plot import get_periodic_x_operator
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products_elementwise,
)

from coherent_rates.system import get_hamiltonian

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        StatisticalValueList,
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

    from coherent_rates.system import PeriodicSystem, PeriodicSystemConfig

_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def _get_isf_pair_states_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    operator: SingleBasisOperator[Any],
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> tuple[StateVectorList[_AX0Inv, Any], StateVectorList[_AX0Inv, Any]]:
    state_evolved = solve_schrodinger_equation_diagonal(
        initial_state,
        times,
        hamiltonian,
    )
    state_evolved_scattered = apply_operator_to_states(operator, state_evolved)

    state_scattered = apply_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation_diagonal(
        state_scattered,
        times,
        hamiltonian,
    )
    return (state_evolved_scattered, state_scattered_evolved)


def get_isf_pair_states(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
    direction: tuple[int] = (1,),
) -> tuple[StateVectorList[_AX0Inv, Any], StateVectorList[_AX0Inv, Any]]:
    operator = get_periodic_x_operator(
        initial_state["basis"],
        direction=direction,
    )
    hamiltonian = get_hamiltonian(system, config)
    return _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )


def _get_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    operator: SingleBasisOperator[Any],
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> ValueList[_AX0Inv]:
    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = _get_isf_pair_states_from_hamiltonian(
        hamiltonian,
        operator,
        initial_state,
        times,
    )

    return calculate_inner_products_elementwise(
        state_scattered_evolved,
        state_evolved_scattered,
    )


def get_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
    direction: tuple[int] = (1,),
) -> ValueList[_AX0Inv]:
    operator = get_periodic_x_operator(
        initial_state["basis"],
        direction=direction,
    )
    hamiltonian = get_hamiltonian(system, config)

    return _get_isf_from_hamiltonian(hamiltonian, operator, initial_state, times)


def _get_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    temperature: float,
    phase: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * Boltzmann * temperature),
    )
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = boltzmann_distribution * np.exp(1j * phase) / normalization
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def _get_random_boltzmann_state_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    temperature: float,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    rng = np.random.default_rng()
    phase = 2 * np.pi * rng.random(len(hamiltonian["data"]))
    return _get_boltzmann_state_from_hamiltonian(hamiltonian, temperature, phase)


def get_random_boltzmann_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> StateVector[ExplicitStackedBasisWithLength[Any, Any]]:
    """Generate a random Boltzmann state.

    Follows the formula described in eqn 5 in
    https://doi.org/10.48550/arXiv.2002.12035.


    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        temperature (float): temperature of the system

    Returns:
    -------
        StateVector[Any]: state with boltzmann distribution

    """
    hamiltonian = get_hamiltonian(system, config)
    return _get_random_boltzmann_state_from_hamiltonian(hamiltonian, config.temperature)


def get_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: _AX0Inv,
    direction: tuple[int] = (1,),
    *,
    n_repeats: int = 10,
) -> ValueList[_AX0Inv]:
    isf_data = np.zeros(times.n, dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    # Convert the operator to the hamiltonian basis
    # to prevent conversion in each repeat
    operator = convert_operator_to_basis(
        get_periodic_x_operator(hamiltonian["basis"][0], direction),
        hamiltonian["basis"],
    )

    for _i in range(n_repeats):
        state = _get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
        )
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data += data["data"]
    return {
        "data": isf_data / n_repeats,
        "basis": times,
    }


def get_boltzmann_isf_stats(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: _AX0Inv,
    direction: tuple[int] = (1,),
    *,
    average_over: int = 10,
) -> StatisticalValueList[_AX0Inv]:
    isf_data = np.zeros((average_over, times.n), dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    operator = get_periodic_x_operator(hamiltonian["basis"][0], direction)

    for _i in range(average_over):
        state = _get_random_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
        )
        data = _get_isf_from_hamiltonian(hamiltonian, operator, state, times)
        isf_data[_i, :] = data["data"]
    mean = np.mean(isf_data, axis=0, dtype=np.complex128)
    sd = np.std(isf_data, axis=0, dtype=np.complex128)
    return {
        "data": mean,
        "basis": times,
        "standard_deviation": sd,
    }
