from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    state_vector_list_into_iter,
)
from surface_potential_analysis.util.plot import get_figure

from coherent_rates.isf import (
    get_ak_data_1d,
    get_alpha_deltak_linear_fit,
    get_boltzmann_isf,
    get_isf_pair_states,
)
from coherent_rates.system import (
    PeriodicSystem,
    PeriodicSystemConfig,
    get_hamiltonian,
    get_potential,
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
    potential = get_potential(system, config)
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
    potential = get_potential(system, config)
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
    potential = get_potential(system, config)
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = ax.twinx()

    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = get_isf_pair_states(system, config, initial_state, times, direction)

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
    times: EvenlySpacedTimeBasis[Any, Any, Any],
    direction: tuple[int] = (1,),
    *,
    n_repeats: int = 10,
) -> None:
    data = get_boltzmann_isf(
        system,
        config,
        times,
        direction,
        n_repeats=n_repeats,
    )
    fig, ax, line = plot_value_list_against_time(data)
    line.set_label("abs ISF")

    fig, ax, line = plot_value_list_against_time(data, ax=ax, measure="real")
    line.set_label("real ISF")

    fig, ax, line = plot_value_list_against_time(data, ax=ax, measure="imag")
    line.set_label("imag ISF")

    ax.set_ylabel("ISF")
    ax.legend()

    fig.show()
    input()


def plot_alpha_deltak(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[int] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
) -> None:
    bound_data = get_ak_data_1d(system, config, nk_points=nk_points, times=times)
    k_points = bound_data["basis"].k_points
    bound_fit = get_alpha_deltak_linear_fit(config, bound_data)
    xfit = np.array([0, k_points[len(k_points) - 1] * 1.2])
    yfit = bound_fit.gradient * xfit + bound_fit.intercept
    fig, ax = get_figure(None)

    ax.plot(k_points, bound_data["data"], "bo", label="Bound")
    ax.plot(xfit, yfit, "b")

    free_system = PeriodicSystem(
        id=system.id,
        barrier_energy=0,
        lattice_constant=system.lattice_constant,
        mass=system.mass,
    )
    free_data = get_ak_data_1d(free_system, config, nk_points=nk_points, times=times)
    k_points_free = free_data["basis"].k_points
    free_fit = get_alpha_deltak_linear_fit(config, free_data)

    xfit1 = np.array([0, k_points_free[len(k_points_free) - 1] * 1.2])
    yfit1 = free_fit.gradient * xfit1 + free_fit.effective_mass

    ax.plot(k_points_free, free_data["data"], "ro", label="Free")
    ax.plot(xfit1, yfit1, "r")
    print(
        "Bound mass =",
        bound_fit.effective_mass,
        "Free mass =",
        free_fit.effective_mass,
    )
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])
    fig.show()
    input()
