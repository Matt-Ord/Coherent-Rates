from __future__ import annotations

from typing import cast

import numpy as np
from matplotlib.axes import Axes
from scipy.constants import Boltzmann, electron_volt, proton_mass  # type: ignore lib
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.state_vector.plot import plot_state_1d_x
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_momentum,
)
from surface_potential_analysis.state_vector.state_vector_list import get_state_vector
from surface_potential_analysis.util.plot import get_figure
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_wannier_states,
)
from surface_potential_analysis.wavepacket.localization._wannier90 import (
    get_localization_operator_wannier90_individual_bands,
)
from surface_potential_analysis.wavepacket.plot import (
    plot_wavepacket_localized_effective_mass_against_energy,
    plot_wavepacket_transformed_energy_1d,
    plot_wavepacket_transformed_energy_effective_mass_against_energy,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
    GaussianMethodWithOffset,
)
from coherent_rates.isf import (
    get_boltzmann_isf,
    get_conditions_at_barrier_energy,
    get_conditions_at_mass,
    get_conditions_at_temperatures,
    get_free_rate_against_momentum,
)
from coherent_rates.plot import (
    plot_effective_mass_against_condition,
    plot_effective_mass_against_momentum,
    plot_effective_mass_against_scattered_energy,
    plot_isf_with_fit,
    plot_rate_against_momentum,
)
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
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

    fig, ax, line_0 = plot_state_1d_x(get_state_vector(states, (20, -1)))
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

    fig, ax, _ = plot_wavepacket_localized_effective_mass_against_energy(
        wavefunctions,
        true_mass=system.mass,
        scale="log",
    )
    ax.set_title("wannier")  # type: ignore library type
    ax.set_xlim(0, 4 * system.barrier_energy)

    line = ax.axvline(system.barrier_energy)  # type: ignore library type
    line.set_color("black")
    line.set_linestyle("--")
    fig.show()


if __name__ == "__main__":
    _effective_mass_demonstration()

    _compare_rate_against_free_surface()
    _compare_rate_against_temperature()
    _plot_isf_below_and_above_barrier_width()
    _compare_small_k_effective_mass_against_temperature()
    _compare_small_k_effective_mass_against_barrier()

    _compare_instrument_sensitivity_against_mass()
    _compare_instrument_sensitivity_against_temperature()

    input()
