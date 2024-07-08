from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib import pyplot as plt
from surface_potential_analysis.operator.operations import apply_operator_to_states
from surface_potential_analysis.operator.operator import (
    apply_operator_to_state,
)
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    get_periodic_x_operator,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)

from coherent_rates.system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_average_boltzmann_isf,
    get_extended_interpolated_potential,
    get_hamiltonian,
    solve_schrodinger_equation,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
    )
    from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
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


def plot_system_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
) -> None:
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, _ax, _anim = animate_state_over_list_1d_x(states)

    fig.show()
    input()


def plot_system_evolution_with_potential(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
) -> None:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = ax.twinx()
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, ax, _anim = animate_state_over_list_1d_x(states, ax=ax1)

    fig.show()
    input()


def plot_pair_system_evolution(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[StackedBasisWithVolumeLike[Any, Any, Any]],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
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
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = ax.twinx()

    fig, ax, _anim1 = animate_state_over_list_1d_x(state_evolved_scattered, ax=ax1)
    fig, ax, _anim2 = animate_state_over_list_1d_x(state_scattered_evolved, ax=ax1)

    fig.show()

    fig, ax = plt.subplots()
    fig, ax, _anim3 = animate_state_over_list_1d_k(state_evolved_scattered, ax=ax)
    fig, ax, _anim4 = animate_state_over_list_1d_k(state_scattered_evolved, ax=ax)

    fig.show()
    input()


def plot_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    temperature: float,
    times: EvenlySpacedTimeBasis[Any, Any, Any],
    direction: tuple[int] = (1,),
    n_repeats: int = 10,
) -> None:
    data = get_average_boltzmann_isf(
        system,
        config,
        times,
        direction,
        temperature,
        n_repeats,
    )
    fig, ax, line = plot_value_list_against_time(data)
    fig.show()
    input()
