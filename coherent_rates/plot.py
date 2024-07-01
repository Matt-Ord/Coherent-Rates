from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.operator.operator import (
    apply_operator_to_state,
    apply_operator_to_states,
)
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
    get_periodic_x_operator,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from coherent_rates.system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_extended_interpolated_potential,
    get_hamiltonian,
    solve_schrodinger_equation,
)

if TYPE_CHECKING:
    from surface_potential_analysis.state_vector.state_vector import StateVector


def plot_system_eigenstates(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    """Plot the potential against position."""
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = hamiltonian["basis"][0].vectors

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(eigenvectors)):
        plot_state_1d_x(state, ax=ax1)

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def plot_system_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> None:
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, ax, _anim = animate_state_over_list_1d_x(states)

    fig.show()
    input()


def plot_pair_system_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
    direction: tuple[int] = (1,),
) -> None:
    operator = get_periodic_x_operator(initial_state["basis"], direction)

    state_evolved = solve_schrodinger_equation(system, config, initial_state, times)

    state_evolved_scattered = apply_operator_to_states(operator, state_evolved)

    state_scattered = apply_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation(
        system,
        config,
        state_scattered,
        times,
    )
    
    fig, ax = plt.subplots()

    fig, ax, _anim1 = animate_state_over_list_1d_x(state_evolved_scattered, ax=ax)
    fig, ax, _anim2 = animate_state_over_list_1d_x(state_scattered_evolved, ax=ax)

    fig.show()

    # fig, ax = plt.subplots()
    # fig, ax, _anim3 = animate_state_over_list_1d_k(state_evolved_scattered, ax=ax)
    # fig, ax, _anim4 = animate_state_over_list_1d_k(state_scattered_evolved, ax=ax)

    # fig.show()
    input()
