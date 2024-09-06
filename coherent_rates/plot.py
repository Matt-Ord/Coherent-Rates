from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence, TypeVar, cast

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.constants import Boltzmann  # type: ignore library type
from surface_potential_analysis.basis.basis_like import BasisLike
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasisWithVolumeLike,
)
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    animate_state_over_list_2d_x,
    plot_state_1d_k,
    plot_state_1d_x,
    plot_state_2d_k,
    plot_state_2d_x,
    plot_total_band_occupation_against_energy,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_all_value_list_against_time,
    plot_split_value_list_against_frequency,
    plot_split_value_list_against_time,
    plot_value_list_against_frequency,
    plot_value_list_against_momentum,
    plot_value_list_against_time,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    get_state_vector,
)
from surface_potential_analysis.util.plot import (
    get_figure,
    plot_data_1d,
)
from surface_potential_analysis.util.squared_scale import SquaredScale
from surface_potential_analysis.util.util import Measure
from surface_potential_analysis.wavepacket.plot import (
    plot_occupation_against_band,
    plot_occupation_against_band_average_energy,
    plot_wavepacket_eigenvalues_1d_k,
    plot_wavepacket_eigenvalues_1d_x,
    plot_wavepacket_eigenvalues_2d_k,
    plot_wavepacket_eigenvalues_2d_x,
    plot_wavepacket_transformed_energy_1d,
    plot_wavepacket_transformed_energy_effective_mass_against_band,
    plot_wavepacket_transformed_energy_effective_mass_against_energy,
)
from surface_potential_analysis.wavepacket.wavepacket import get_wavepacket_at_band

from coherent_rates.fit import GaussianMethod, get_default_isf_times
from coherent_rates.isf import (
    SimulationCondition,
    get_analytical_isf,
    get_band_resolved_boltzmann_isf,
    get_boltzmann_isf,
    get_conditions_at_directions,
    get_conditions_at_mass,
    get_conditions_at_temperatures,
    get_effective_mass_against_condition_data,
    get_isf_pair_states,
    get_rate_against_momentum_data,
    get_rate_against_momentum_linear_fit,
    get_scattered_energy_change_against_k,
    get_thermal_scattered_energy_change_against_k,
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
from coherent_rates.state import (
    get_boltzmann_state_from_hamiltonian,
    get_random_boltzmann_state,
)
from coherent_rates.system import (
    FreeSystem,
    PeriodicSystem,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.basis.momentum_basis_like import MomentumBasis
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisWithVolumeLike,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import ValueList
    from surface_potential_analysis.types import SingleIndexLike

    from coherent_rates.config import PeriodicSystemConfig
    from coherent_rates.fit import FitMethod

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])

    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def plot_system_eigenstates_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    states: Iterable[int] | None = None,
) -> None:
    """Plot the potential against position."""
    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, _ = plot_potential_1d_x(potential)

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = hamiltonian["basis"][0].vectors

    ax1 = cast(Axes, ax.twinx())
    fig2, ax2 = plt.subplots()  # type: ignore library type
    states = range(3) if states is None else states
    for idx in states:
        state = get_state_vector(eigenvectors, idx=idx)
        plot_state_1d_x(state, ax=ax1)

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_system_eigenstates_2d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    states: Iterable[int] | None = None,
    bloch_k: SingleIndexLike | None = None,
) -> None:
    hamiltonian = get_hamiltonian(system, config)

    bloch_k = 0 if bloch_k is None else bloch_k
    states = range(config.n_bands) if states is None else states

    eigenvectors = hamiltonian["basis"][0].vectors_at_bloch_k(bloch_k)

    for i in states:
        state = get_state_vector(eigenvectors, i)
        fig, _ax, _line = plot_state_2d_x(state)
        fig.show()
        fig, _ax, _line = plot_state_2d_k(state)
        fig.show()

    input()


def plot_system_eigenvalues(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    wavefunctions = get_bloch_wavefunctions(system, config)

    fig, _ = plot_wavepacket_eigenvalues_1d_k(wavefunctions)
    fig.show()

    fig, _ = plot_wavepacket_eigenvalues_1d_x(wavefunctions)
    fig.show()

    if len(config.shape) > 1:
        wavepacket_0 = get_wavepacket_at_band(wavefunctions, 0)
        fig, _, _ = plot_wavepacket_eigenvalues_2d_k(wavepacket_0)
        fig.show()

        fig, _, _ = plot_wavepacket_eigenvalues_2d_x(wavepacket_0)
        fig.show()

    fig, ax, _ = plot_wavepacket_transformed_energy_1d(
        wavefunctions,
        free_mass=system.mass,
        measure="abs",
    )
    ax.set_title(  # type: ignore library type
        "Plot of the lowest component of the fourier transform of energy"
        "\nagainst that of a free particle",
    )
    ax.legend()  # type: ignore library type
    fig.show()


def plot_system_bands(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    """Investigate the Bandstructure of a system."""
    wavefunctions = get_bloch_wavefunctions(system, config)

    fig, ax, line0 = plot_wavepacket_transformed_energy_effective_mass_against_band(
        wavefunctions,
    )

    ax.set_ylim(0, 4 * system.mass)
    _, _, line1 = plot_occupation_against_band(
        wavefunctions,
        config.temperature,
        ax=cast(Axes, ax.twinx()),
    )
    line1.set_color("C1")
    ax.legend(handles=[line0, line1])  # type: ignore library type
    fig.show()

    fig, ax, line0 = plot_wavepacket_transformed_energy_effective_mass_against_energy(
        wavefunctions,
    )
    _, _, line1 = plot_occupation_against_band_average_energy(
        wavefunctions,
        config.temperature,
        ax=cast(Axes, ax.twinx()),
    )
    line1.set_color("C1")

    ax.set_ylim(0, 4 * system.mass)
    line = ax.axvline(system.barrier_energy)  # type: ignore library type
    line.set_color("black")
    line.set_linestyle("--")
    ax.legend(handles=[line0, line1])  # type: ignore library type
    fig.show()
    input()


def plot_system_evolution_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_B0],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
) -> None:
    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = cast(Axes, ax.twinx())
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, ax, _anim = animate_state_over_list_1d_x(states, ax=ax1)

    fig.show()
    input()


def plot_system_evolution_2d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
) -> None:
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, _ax, _anim = animate_state_over_list_2d_x(states)

    fig.show()
    input()


def plot_pair_system_evolution_1d(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: EvenlySpacedTimeBasis[Any, Any, Any],
    initial_state: StateVector[_SBV0]
    | StateVector[ExplicitStackedBasisWithLength[Any, Any]]
    | None = None,
    *,
    measure: Measure = "abs",
) -> None:
    initial_state = (
        get_random_boltzmann_state(system, config)
        if initial_state is None
        else initial_state
    )
    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = cast(Axes, ax.twinx())

    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = get_isf_pair_states(system, config, initial_state, times)

    fig, ax, _anim1 = animate_state_over_list_1d_x(
        state_evolved_scattered,
        ax=ax1,
        measure=measure,
    )
    fig, ax, _anim2 = animate_state_over_list_1d_x(
        state_scattered_evolved,
        ax=ax1,
        measure=measure,
    )

    fig.show()

    fig, ax = plt.subplots()  # type: ignore library type
    fig, ax, _anim3 = animate_state_over_list_1d_k(
        state_evolved_scattered,
        ax=ax,
        measure=measure,
    )
    fig, ax, _anim4 = animate_state_over_list_1d_k(
        state_scattered_evolved,
        ax=ax,
        measure=measure,
    )

    fig.show()
    input()


_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])


def plot_isf_with_fit(
    data: ValueList[_BT0],
    method: FitMethod[Any],
    *,
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> tuple[Figure, Axes]:
    fig, ax, line = plot_value_list_against_time(data)
    line.set_label("ISF")

    fit = method.get_fit_from_isf(
        data,
        system=system,
        config=config,
    )
    fitted_data = method.get_fitted_data(fit, data["basis"])

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax)
    line.set_label("Fit")

    ax.legend()  # type: ignore bad types
    return (fig, ax)


def plot_isf_fit_for_conditions(
    conditions: list[SimulationCondition],
    *,
    fit_method: FitMethod[Any] | None = None,
) -> None:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    for system, config, label in conditions:
        isf = get_boltzmann_isf(
            system,
            config,
            fit_method.get_fit_times(system=system, config=config),
        )
        fig, ax = plot_isf_with_fit(isf, fit_method, system=system, config=config)
        ax.set_title(f"ISF with fit for {label}")  # type: ignore unknown
        fig.show()

    input()


def plot_rate_against_momentum_isf_fit(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]],
) -> None:
    plot_isf_fit_for_conditions(
        get_conditions_at_directions(system, config, directions),
        fit_method=fit_method,
    )


def plot_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
    *,
    n_repeats: int = 10,
) -> None:
    times = (
        get_default_isf_times(system=system, config=config) if times is None else times
    )
    data = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )

    fig, ax, line = plot_value_list_against_time(data)
    line.set_label("abs ISF")

    fig, ax, line = plot_value_list_against_time(data, ax=ax, measure="real")
    line.set_label("real ISF")

    fig, ax, line = plot_value_list_against_time(data, ax=ax, measure="imag")
    line.set_label("imag ISF")
    ax.legend()  # type: ignore library type

    ax.set_title("Plot of the ISF against time")  # type: ignore library type

    fig.show()

    fig, ax, line = plot_value_list_against_frequency(data)
    line.set_label("abs ISF")
    fig, ax, line = plot_value_list_against_frequency(data, measure="imag", ax=ax)
    line.set_label("imag ISF")
    fig, ax, line = plot_value_list_against_frequency(data, measure="real", ax=ax)
    line.set_label("real ISF")
    ax.legend()  # type: ignore library type
    ax.set_title("Plot of the fourier transform of the ISF against time")  # type: ignore library type
    fig.show()

    input()


def plot_free_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
    *,
    n_repeats: int = 10,
) -> None:
    system = FreeSystem(system)
    times = (
        get_default_isf_times(system=system, config=config) if times is None else times
    )
    isf = get_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )
    analytical_isf = get_analytical_isf(system, config, times)

    measures = list[Measure](["real", "imag", "abs"])
    for measure in measures:
        fig, ax, line = plot_value_list_against_time(isf)
        line.set_label("Simulated")
        fig, ax, line = plot_value_list_against_time(
            analytical_isf,
            ax=ax,
            measure=measure,
        )
        line.set_label("Analytical")
        line.set_linestyle("--")
        ax.set_title(f"Free ISF ({measure})")  # type: ignore library type
        ax.legend()  # type: ignore library type
        fig.show()

    input()


def plot_band_resolved_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
    *,
    n_repeats: int = 10,
) -> None:
    times = (
        get_default_isf_times(system=system, config=config) if times is None else times
    )

    resolved_data = get_band_resolved_boltzmann_isf(
        system,
        config,
        times,
        n_repeats=n_repeats,
    )
    fig, _ = plot_split_value_list_against_time(resolved_data, measure="real")
    fig, _ = plot_all_value_list_against_time(resolved_data, measure="real")

    fig.show()
    fig, ax = plot_split_value_list_against_frequency(resolved_data)
    ax.set_title("Plot of the fourier transform of the ISF against time")  # type: ignore library type
    fig.show()
    input()


def _plot_rate_against_momentum(
    data: ValueList[MomentumBasis],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax, line = plot_value_list_against_momentum(data, ax=ax)
    line.set_linestyle("")
    line.set_marker("x")

    fit = get_rate_against_momentum_linear_fit(data)

    k_points = data["basis"].k_points
    x_fit = np.array([0, k_points[-1] * 1.2])
    y_fit = fit.gradient * x_fit + fit.intercept
    _, _, fit_line = plot_data_1d(y_fit, x_fit, ax=ax, measure="real")  # type: ignore library type
    fit_line.set_color(line.get_color())

    ax.set_xlabel("delta k /$m^{-1}$")  # type: ignore library type
    ax.set_ylabel("rate")  # type: ignore library type

    return fig, ax, line


def plot_rate_against_momentum(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    data = get_rate_against_momentum_data(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
    )

    fig, ax, line = _plot_rate_against_momentum(data, ax=ax)
    line.set_label(fit_method.get_rate_label())

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.legend()  # type: ignore library type
    ax.set_title("Plot of rate against delta k")  # type: ignore library type

    return (fig, ax, line)


def plot_effective_mass_against_condition(
    conditions: list[tuple[PeriodicSystem, PeriodicSystemConfig, str]],
    x_values: np.ndarray[Any, np.dtype[np.float64]],
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    temperature_ax: Axes | None = None,
) -> tuple[tuple[Figure, Axes, Line2D], tuple[Figure, Axes]]:
    fit_method = GaussianMethod() if fit_method is None else fit_method

    momentum_plot = get_figure(None)
    for system, config, label in conditions:
        _, _, line = plot_rate_against_momentum(
            system,
            config,
            directions=directions,
            fit_method=fit_method,
            ax=momentum_plot[1],
        )
        line.set_label(label)
    momentum_plot[1].legend()  # type: ignore unknown

    data = get_effective_mass_against_condition_data(
        conditions,
        fit_method=fit_method,
        directions=directions,
    )

    fig, temperature_ax, line = plot_data_1d(
        data["data"],
        x_values,
        ax=temperature_ax,
    )
    temperature_ax.set_ylabel("Effective Mass /kg")  # type: ignore unknown
    return ((fig, temperature_ax, line), momentum_plot)


def plot_effective_mass_against_mass(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    masses: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> None:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    masses = (
        np.array([(1 + 5 * i) * system.mass for i in range(5)])
        if masses is None
        else masses
    )

    conditions = get_conditions_at_mass(system, config, masses)

    mass_plot, momentum_plot = plot_effective_mass_against_condition(
        conditions,
        masses,
        fit_method=fit_method,
        directions=directions,
    )

    momentum_plot[0].show()

    fig, ax, line = mass_plot

    ax.set_xlabel("Mass /kg")  # type: ignore unknown
    ax.set_title(  # type: ignore unknown
        "Plot of Effective mass against mass for"
        f" {fit_method.get_rate_label()} rate",
    )
    _, _, line = plot_data_1d(masses, masses, ax=ax)
    line.set_color("black")
    line.set_linestyle("--")
    fig.show()
    input()


def plot_effective_mass_against_temperature(  # noqa: PLR0913
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    temperatures: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    directions: list[tuple[int, ...]] | None = None,
    fit_method: FitMethod[Any] | None = None,
    temperature_ax: Axes | None = None,
) -> tuple[tuple[Figure, Axes, Line2D], tuple[Figure, Axes]]:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    temperatures = (
        np.array([(60 + 30 * i) for i in range(5)])
        if temperatures is None
        else temperatures
    )

    conditions = get_conditions_at_temperatures(system, config, temperatures)

    temperature_plot, momentum_plot = plot_effective_mass_against_condition(
        conditions,
        temperatures,
        fit_method=fit_method,
        directions=directions,
        temperature_ax=temperature_ax,
    )

    _, temperature_ax, _ = temperature_plot

    temperature_ax.set_title(  # type: ignore unknown
        "Plot of Effective mass against temperature for"
        f" {fit_method.get_rate_label()} rate",
    )
    temperature_ax.axhline(system.mass, color="black", ls="--")  # type: ignore unknown
    temperature_ax.set_ylim(0, max(1.2 * system.mass, temperature_ax.get_ylim()[1]))
    temperature_ax.set_xlabel("Temperature / k")  # type: ignore unknown

    return temperature_plot, momentum_plot


def plot_barrier_temperature(
    barrier_energy: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Line2D]:
    fig, ax = get_figure(ax)

    barrier_temperature = barrier_energy / Boltzmann
    line = ax.axvline(barrier_temperature, ls="--")  # type: ignore unknown
    line.set_label("Barrier Height")
    return fig, ax, line


def plot_effective_mass_against_temperature_comparison(
    conditions: Sequence[SimulationCondition],
    *,
    fit_method: FitMethod[Any] | None = None,
    temperatures: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> None:
    fig, ax = get_figure(None)

    for system, config, label in conditions:
        (_, _, line), momentum_plot = plot_effective_mass_against_temperature(
            system,
            config,
            fit_method=fit_method,
            temperatures=temperatures,
            directions=directions,
            temperature_ax=ax,
        )
        momentum_plot[1].set_title(f"Plot of rate against delta k at {label}")  # type: ignore unknown
        momentum_plot[0].show()
        line.set_label(label)

    ax.legend()  # type: ignore unknown
    fig.show()
    input()


def plot_thermal_scattered_energy_change_comparison(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    directions: list[tuple[int, ...]] | None = None,
    n_repeats: int = 10,
) -> None:
    bound_data = get_thermal_scattered_energy_change_against_k(
        system,
        config,
        directions=directions,
        n_repeats=n_repeats,
    )
    fig, ax, line = plot_value_list_against_momentum(bound_data)
    line.set_label("Bound")

    free_system = FreeSystem(system)
    free_data = get_thermal_scattered_energy_change_against_k(
        free_system,
        config,
        directions=directions,
        n_repeats=1,
    )
    fig, ax, line1 = plot_value_list_against_momentum(free_data, ax=ax)
    line1.set_label("Free")

    ax.legend()  # type: ignore library type
    ax.set_xscale(SquaredScale(axis=None))  # type: ignore library type
    ax.set_ylabel("Energy change /J")  # type: ignore library type

    fig.show()
    input()


def plot_scattered_energy_change_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    state: StateVector[Any],
    *,
    directions: list[tuple[int, ...]] | None = None,
) -> None:
    bound_data = get_scattered_energy_change_against_k(
        system,
        config,
        state,
        directions=directions,
    )
    fig, ax, _ = plot_value_list_against_momentum(bound_data)
    ax.set_xscale(SquaredScale(axis=None))  # type: ignore library type
    ax.set_title("Quadratic")  # type: ignore library type
    ax.set_ylabel("Energy change /J")  # type: ignore library type
    fig.show()

    fig, ax, _ = plot_value_list_against_momentum(bound_data)
    ax.set_title("Linear")  # type: ignore library type
    ax.set_ylabel("Energy change /J")  # type: ignore library type
    fig.show()

    input()


def plot_occupation_against_energy_change_with_contition(
    conditions: list[SimulationCondition],
) -> tuple[Figure, Axes]:
    rng = np.random.default_rng()

    fig, ax = get_figure(None)

    for system, config, label in conditions:
        hamiltonian = get_hamiltonian(system, config)
        phase = 2 * np.pi * rng.random(len(hamiltonian["data"]))

        state = get_boltzmann_state_from_hamiltonian(
            hamiltonian,
            config.temperature,
            phase,
        )
        operator = get_instrument_biased_periodic_x(
            hamiltonian,
            config.direction,
            config.scattered_energy_range,
        )
        scattered_state = apply_scattering_operator_to_state(operator, state)

        fig, ax, line = plot_total_band_occupation_against_energy(
            hamiltonian,
            scattered_state,
            ax=ax,
        )
        line.set_label(label)

    ax.legend()  # type: ignore library type
    return fig, ax


def plot_occupation_against_energy_change_comparison_mass(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    mass_ratio: float,
) -> None:
    conditions = [
        (system, config, "Normal Mass"),
        (
            system.with_mass(mass_ratio * system.mass),
            config,
            f"{mass_ratio}$\\times$ mass",
        ),
    ]

    fig, ax = plot_occupation_against_energy_change_with_contition(
        conditions,
    )

    ax.axvline(system.barrier_energy, color="black", ls="--")  # type: ignore library type

    ax.set_xlim(0, 10 * system.barrier_energy)
    ax.set_ylim(0)
    ax.legend()  # type: ignore library type
    fig.show()
    input()


def plot_occupation_against_energy_change_comparison_temperature(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    temperatures: tuple[float, float],
) -> None:
    conditions = get_conditions_at_temperatures(system, config, temperatures)
    fig, ax = plot_occupation_against_energy_change_with_contition(
        conditions,
    )

    line = ax.axvline(system.barrier_energy, color="black", ls="--")  # type: ignore library type
    line.set_label("Barrier Energy")
    ax.set_xlim(0, 10 * system.barrier_energy)
    ax.set_ylim(0)
    ax.legend()  # type: ignore library type
    fig.show()
    input()
