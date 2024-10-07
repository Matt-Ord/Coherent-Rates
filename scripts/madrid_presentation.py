"""Script to generate the plots required for the Madrid presentation."""

from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib.axes import Axes
from scipy.constants import Boltzmann, electron_volt, proton_mass  # type: ignore lib
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_to_basis,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_2d_x,
    plot_state_1d_x,
    plot_state_2d_k,
    plot_state_2d_x,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_momentum,
    plot_value_list_against_time,
)
from surface_potential_analysis.state_vector.state_vector_list import get_state_vector
from surface_potential_analysis.util.plot import get_axis_colorbar, get_figure
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_wannier_states,
)
from surface_potential_analysis.wavepacket.localization._wannier90 import (
    get_localization_operator_wannier90_individual_bands,
)
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_eigenvalues_1d_k,
    plot_wavepacket_transformed_energy_1d,
    plot_wavepacket_transformed_energy_effective_mass_against_energy,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
    GaussianMethodWithOffset,
    GaussianPlusExponentialMethod,
    get_free_particle_time,
)
from coherent_rates.isf import (
    get_boltzmann_isf,
    get_conditions_at_barrier_energy,
    get_conditions_at_mass,
    get_conditions_at_temperatures,
    get_free_rate_against_momentum,
)
from coherent_rates.plot import (
    plot_band_resolved_boltzmann_isf,
    plot_effective_mass_against_condition,
    plot_effective_mass_against_momentum,
    plot_effective_mass_against_scattered_energy,
    plot_free_isf_comparison,
    plot_isf_with_fit,
    plot_rate_against_momentum,
)
from coherent_rates.scattering_operator import (
    apply_scattering_operator_to_state,
    get_instrument_biased_periodic_x,
)
from coherent_rates.solve import (
    get_bloch_wavefunctions,
    get_hamiltonian,
    solve_schrodinger_equation,
)
from coherent_rates.state import get_coherent_state
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
    FreeSystem,
)


def _compare_rate_against_free_surface() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=50,
        temperature=100,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    directions = [(i,) for i in [1, 2, *list(range(5, 105, 5))]]

    fig, ax = get_figure(None)

    _, _, line = plot_rate_against_momentum(
        system,
        config,
        fit_method=GaussianMethod(),
        directions=directions,
        ax=ax,
    )
    line.set_label("Bound system")

    _, _, line = plot_rate_against_momentum(
        FreeSystem(system),
        config,
        fit_method=GaussianMethod(),
        directions=directions,
        ax=ax,
    )
    line.set_label("Free system")

    _, _, free_line = plot_value_list_against_momentum(
        get_free_rate_against_momentum(system, config, directions=directions),
        ax=ax,
    )
    free_line.set_color(line.get_color())

    ax.legend()  # type: ignore library type

    fig.show()


def _compare_rate_against_temperature() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=30,
        temperature=100,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    conditions = get_conditions_at_temperatures(
        system,
        config,
        [100, 120, 140, 160, 180, 200, 220, 240],
    )

    fig, ax = get_figure(None)
    directions = [(i,) for i in [1, 2, *list(range(5, 105, 5))]]
    for system, config, label in conditions:
        _, _, line = plot_effective_mass_against_momentum(
            system,
            config,
            fit_method=GaussianMethod(),
            directions=directions,
            ax=ax,
        )
        line.set_label(label)
    line = ax.axvline(1 / system.lattice_constant)  # type: ignore lib
    line.set_linestyle("--")
    line.set_color("black")
    line.set_alpha(0.8)
    line.set_label("Barrier Width")

    ax.legend()  # type: ignore lib
    fig.show()

    fig, ax = get_figure(None)
    directions = [(i,) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    for system, config, label in conditions:
        _, _, line = plot_effective_mass_against_momentum(
            system,
            config,
            fit_method=GaussianMethod(),
            directions=directions,
            ax=ax,
        )
        line.set_label(label)
    line = ax.axvline(1 / system.lattice_constant)  # type: ignore lib
    line.set_linestyle("--")
    line.set_color("black")
    line.set_alpha(0.8)

    ax.legend()  # type: ignore lib
    fig.show()


def _plot_isf_below_and_above_barrier_width() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=30,
        temperature=100,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    method = GaussianMethod()

    config = config.with_direction((2,)).with_shape((1000,))
    times = method.get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)
    fig, _ax = plot_isf_with_fit(
        isf,
        method,
        system=system,
        config=config,
    )
    fig.show()

    config = config.with_temperature(240)
    times = method.get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)
    fig, _ax = plot_isf_with_fit(
        isf,
        method,
        system=system,
        config=config,
    )
    fig.show()

    config = config.with_direction((50,)).with_shape((100,)).with_temperature(100)
    times = method.get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)
    fig, _ax = plot_isf_with_fit(
        isf,
        method,
        system=system,
        config=config,
    )
    fig.show()

    config = config.with_direction((50,)).with_shape((100,)).with_temperature(100)
    fig, _ax = plot_isf_with_fit(
        isf,
        method,
        system=system.with_barrier_energy(0.0),
        config=config,
    )
    fig.show()

    plot_band_resolved_boltzmann_isf(
        system,
        config,
        GaussianMethod().get_fit_times(system=system, config=config),
    )

    config = config.with_direction((20,)).with_shape((100,))
    times = EvenlySpacedTimeBasis(
        200,
        1,
        0,
        4 * get_free_particle_time(system=system, config=config),
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)
    fig, _ax, _line = plot_value_list_against_time(isf)
    fig.show()


def _compare_small_k_effective_mass_against_temperature() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=60,
        temperature=100,
        direction=(5,),
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    temperatures = np.array([100, 120, 140, 160, 180, 200, 220, 240, 260])
    conditions = get_conditions_at_temperatures(
        system,
        config,
        temperatures,
    )
    fig, _ax, _ = plot_effective_mass_against_condition(conditions, temperatures)
    fig.show()


def _compare_small_k_effective_mass_against_barrier() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=60,
        temperature=100,
        direction=(5,),
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    ratios = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2])
    conditions = get_conditions_at_barrier_energy(
        system,
        config,
        [Boltzmann * config.temperature * r for r in ratios],
    )
    fig, _, _ = plot_effective_mass_against_condition(conditions, 1 / ratios)  # type: ignore lib
    fig.show()


def _compare_instrument_sensitivity_against_mass() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=50,
        temperature=100,
        scattered_energy_range=(-0.005 * electron_volt, 0.005 * electron_volt),
    )
    system = FreeSystem(SODIUM_COPPER_BRIDGE_SYSTEM_1D)
    directions = [(i,) for i in [1, 2, *list(range(5, 105, 5))]]

    fig, ax = get_figure(None)
    conditions = get_conditions_at_mass(
        system,
        config,
        [i * proton_mass for i in (5, 10, 15, 20, 30, 40)],
    )
    for system, config, label in conditions:
        _, _, line = plot_effective_mass_against_scattered_energy(
            system,
            config,
            fit_method=GaussianMethodWithOffset(),
            directions=directions,
            ax=ax,
        )
        line.set_label(label)

    ax.legend()  # type: ignore lib
    fig.show()

    fig, ax = get_figure(None)

    for system, config, label in conditions:
        _, _, line = plot_effective_mass_against_momentum(
            system,
            config,
            fit_method=GaussianMethodWithOffset(),
            directions=directions,
            ax=ax,
        )
        line.set_label(label)

    line = ax.axvline(1 / system.lattice_constant)  # type: ignore lib
    line.set_linestyle("--")
    line.set_color("black")
    line.set_alpha(0.8)

    ax.legend()  # type: ignore lib
    fig.show()


def _compare_instrument_sensitivity_against_temperature() -> None:
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=50,
        temperature=100,
        scattered_energy_range=(-0.005 * electron_volt, 0.005 * electron_volt),
    )
    system = FreeSystem(SODIUM_COPPER_BRIDGE_SYSTEM_1D)
    directions = [(i,) for i in [1, 2, *list(range(5, 105, 5))]]

    conditions = get_conditions_at_temperatures(
        system,
        config,
        [50, 100, 150, 200],
    )
    fig, ax = get_figure(None)
    for system, config, label in conditions:
        _, _, line = plot_effective_mass_against_scattered_energy(
            system,
            config,
            fit_method=GaussianMethodWithOffset(),
            directions=directions,
            ax=ax,
        )
        line.set_label(label)

    ax.legend()  # type: ignore lib
    fig.show()


def _effective_mass_demonstration() -> None:
    config = PeriodicSystemConfig(
        (200,),
        (100,),
        truncation=50,
        temperature=100,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    wavefunctions = get_bloch_wavefunctions(system, config)

    fig, _, lines = plot_wavepacket_eigenvalues_1d_k(
        wavefunctions,
        bands=list(range(10)),
    )

    for line in lines:
        line.set_marker("")
        line.set_linestyle("-")
    fig.show()

    fig, _ax, _ = plot_wavepacket_transformed_energy_1d(
        wavefunctions,
        free_mass=system.mass,
        measure="abs",
    )
    fig.show()

    fig, ax, _ = plot_wavepacket_transformed_energy_effective_mass_against_energy(
        wavefunctions,
    )
    ax.set_xlim(0, 4 * system.barrier_energy)
    ax.set_ylim(0, 3 * system.mass)
    line = ax.axvline(system.barrier_energy)  # type: ignore library type
    line.set_color("black")
    line.set_linestyle("--")
    fig.show()

    fig, ax, _ = plot_wavepacket_transformed_energy_effective_mass_against_energy(
        wavefunctions,
        true_mass=system.mass,
        scale="log",
    )
    ax.set_xlim(0, 4 * system.barrier_energy)

    line = ax.axvline(system.barrier_energy)  # type: ignore library type
    line.set_color("black")
    line.set_linestyle("--")
    fig.show()

    config = config.with_shape((10,))
    wavefunctions = get_bloch_wavefunctions(system, config)

    operator = get_localization_operator_wannier90_individual_bands(
        wavefunctions,
    )
    states = get_wannier_states(wavefunctions, operator)

    fig, ax, line_0 = plot_state_1d_x(get_state_vector(states, (0, -1)))
    _, _, line_1 = plot_state_1d_x(get_state_vector(states, (0, -2)), ax=ax)
    line_1.set_linestyle("--")
    line_1.set_color(line_0.get_color())

    _, _, line_2 = plot_potential_1d_x(
        system.get_potential(config.shape, config.resolution),
        ax=cast(Axes, ax.twinx()),
    )
    line_2.set_color("black")
    line_2.set_alpha(0.1)
    line_2.set_linewidth(4)

    ax.set_xlim(0, 3 * system.lattice_constant)
    fig.show()

    config = config.with_shape((40,))
    wavefunctions = get_bloch_wavefunctions(system, config)


def _2d_state_evolved_scattered_demo() -> None:
    config = PeriodicSystemConfig(
        (3, 3),
        (30, 30),
        truncation=150,
        direction=(30, 15),
        temperature=50,
    )
    system = SODIUM_COPPER_SYSTEM_2D
    sigma_0 = (system.lattice_constant / 12, system.lattice_constant / 12)

    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)

    origin = (4 / 3) * (
        hamiltonian["basis"][0].delta_x_stacked[0]
        + hamiltonian["basis"][0].delta_x_stacked[1]
    )
    initial_state = get_coherent_state(
        hamiltonian["basis"][0],
        tuple(origin),
        (0, 0),
        sigma_0,
    )

    fig, _, _ = plot_state_2d_k(initial_state)
    fig.show()
    fig, _, _ = plot_state_2d_x(initial_state)
    fig.show()

    fig, _, _ = plot_state_2d_x(
        convert_state_vector_to_basis(
            initial_state,
            get_hamiltonian.load_or_call_cached(system, config)["basis"][0],
        ),
    )
    fig.show()
    times = GaussianPlusExponentialMethod("Exponential").get_fit_times(
        system=system,
        config=config,
    )

    states = solve_schrodinger_equation(system, config, initial_state, times)
    fig, ax, _anim0 = animate_state_over_list_2d_x(states)
    if (cb := get_axis_colorbar(ax)) is not None:
        cb.remove()
    _anim0.save("/workspaces/coherent_rates/out0.mp4")
    fig.show()

    hamiltonian = get_hamiltonian.load_or_call_cached(system, config)
    scattering_operator = get_instrument_biased_periodic_x(
        hamiltonian,
        config.direction,
        config.scattered_energy_range,
    )
    scattered_state = apply_scattering_operator_to_state(
        scattering_operator,
        initial_state,
    )

    fig, ax, _ = plot_state_2d_k(scattered_state)
    ax.set_title("scattered_state k")  # type: ignore unknown
    fig.show()
    fig, ax, _ = plot_state_2d_x(scattered_state)
    ax.set_title("scattered_state x")  # type: ignore unknown
    fig.show()

    states = solve_schrodinger_equation(system, config, scattered_state, times)
    fig, ax, _anim1 = animate_state_over_list_2d_x(states)
    if (cb := get_axis_colorbar(ax)) is not None:
        cb.remove()
    _anim1.save("/workspaces/coherent_rates/out1.mp4")
    fig.show()


def _free_isf_demo() -> None:
    config = PeriodicSystemConfig(
        (200,),
        (100,),
        direction=(1,),
        truncation=50,
        temperature=100,
    )
    system = FreeSystem(SODIUM_COPPER_BRIDGE_SYSTEM_1D)

    plot_free_isf_comparison(system, config)
    plot_band_resolved_boltzmann_isf(system, config)


if __name__ == "__main__":
    _effective_mass_demonstration()
    _2d_state_evolved_scattered_demo()
    _free_isf_demo()

    _compare_rate_against_free_surface()
    _compare_rate_against_temperature()
    _plot_isf_below_and_above_barrier_width()
    _compare_small_k_effective_mass_against_temperature()
    _compare_small_k_effective_mass_against_barrier()

    _compare_instrument_sensitivity_against_mass()
    _compare_instrument_sensitivity_against_temperature()

    input()
