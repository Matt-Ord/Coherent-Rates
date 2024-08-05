from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.potential.plot import (
    plot_potential_1d_x,
    plot_potential_2d_x,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_k,
    animate_state_over_list_1d_x,
    animate_state_over_list_2d_x,
    plot_state_1d_k,
    plot_state_1d_x,
    plot_state_2d_k,
    plot_state_2d_x,
)
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_split_value_list_against_frequency,
    plot_split_value_list_against_time,
    plot_value_list_against_frequency,
    plot_value_list_against_nx,
    plot_value_list_against_time,
)
from surface_potential_analysis.state_vector.state_vector_list import (
    get_state_vector,
    state_vector_list_into_iter,
)
from surface_potential_analysis.util.plot import Scale, get_figure
from surface_potential_analysis.wavepacket.plot import (
    get_wavepacket_effective_mass,
    plot_occupation_against_band,
    plot_wavepacket_eigenvalues_1d_k,
    plot_wavepacket_eigenvalues_1d_x,
    plot_wavepacket_transformed_energy_1d,
    plot_wavepacket_transformed_energy_effective_mass_1d,
)

from coherent_rates.isf import (
    MomentumBasis,
    get_ak_data,
    get_band_resolved_boltzmann_isf,
    get_boltzmann_isf,
    get_free_particle_time,
    get_isf_pair_states,
    get_scattered_energy_change_against_k,
)
from coherent_rates.system import (
    FreeSystem,
    PeriodicSystem,
    PeriodicSystem1d,
    PeriodicSystem2d,
    PeriodicSystemConfig,
    get_bloch_wavefunctions,
    get_hamiltonian,
    solve_schrodinger_equation,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from surface_potential_analysis.basis.basis_like import BasisLike
    from surface_potential_analysis.basis.stacked_basis import (
        StackedBasisLike,
        StackedBasisWithVolumeLike,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import ValueList
    from surface_potential_analysis.types import SingleFlatIndexLike
    from surface_potential_analysis.util.util import Measure
    from surface_potential_analysis.wavepacket.wavepacket import (
        BlochWavefunctionListWithEigenvaluesList,
    )

    _B0 = TypeVar("_B0", bound=BasisLike[Any, Any])

    _SB0 = TypeVar("_SB0", bound=StackedBasisLike[Any, Any, Any])
    _SBV0 = TypeVar("_SBV0", bound=StackedBasisWithVolumeLike[Any, Any, Any])


def plot_system_eigenstates_1d(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
) -> None:
    """Plot the potential against position."""
    potential = system.get_potential(config.shape, config.resolution)
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


def _get_effective_mass_rates(
    mass: ValueList[_B0],
    temperature: float,
) -> ValueList[_B0]:
    """Calculate classical rate using equipartition of energies.

    sigma = sqrt(2 m) / (delta_k * sqrt(2 k t))
    rate = sqrt(kt / m) * delta_k

    The rate given here is in units of scattered k (= sqrt(kt / m))
    """
    return {
        "basis": mass["basis"],
        "data": np.sqrt(Boltzmann * temperature / mass["data"]),
    }


def plot_wavepacket_transformed_energy_rate(  # noqa: PLR0913
    wavepacket: BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ],
    temperature: float,
    axes: tuple[int,] = (0,),
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, Line2D]:
    """Plot the energy of the eigenstates in a wavepacket.

    Parameters
    ----------
    wavepacket : BlochWavefunctionListWithEigenvaluesList[
        _B0,
        _SB0,
        _SBV0,
    ]
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]

    """
    fig, ax, line = plot_value_list_against_nx(
        _get_effective_mass_rates(
            get_wavepacket_effective_mass(wavepacket, axes[0]),
            temperature,
        ),
        ax=ax,
        scale=scale,
        measure=measure,
    )
    line.set_label("Thermal Rate")

    ax.set_xlabel("Band Index")
    ax.set_ylabel("Rate / s^-1")
    ax.set_ylim((0.0, ax.get_ylim()[1]))

    return fig, ax, line


def plot_system_bands(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> None:
    """Investigate the Bandstructure of a system."""
    wavefunctions = get_bloch_wavefunctions(system, config)

    fig, _ = plot_wavepacket_eigenvalues_1d_k(wavefunctions)
    fig.show()

    fig, _ = plot_wavepacket_eigenvalues_1d_x(wavefunctions)
    fig.show()

    fig, ax, _ = plot_wavepacket_transformed_energy_1d(
        wavefunctions,
        free_mass=system.mass,
        measure="abs",
    )
    ax.legend()
    fig.show()

    fig, _, _ = plot_wavepacket_transformed_energy_effective_mass_1d(
        wavefunctions,
    )
    fig.show()

    fig, ax, line0 = plot_wavepacket_transformed_energy_rate(
        wavefunctions,
        config.temperature,
    )
    _, _, line1 = plot_occupation_against_band(
        wavefunctions,
        config.temperature,
        ax=ax.twinx(),
    )
    line1.set_color("C1")
    ax.legend(handles=[line0, line1])
    fig.show()
    input()


def plot_system_eigenstates_2d(
    system: PeriodicSystem2d,
    config: PeriodicSystemConfig,
    index: SingleFlatIndexLike | None = None,
) -> None:
    """Plot the potential against position."""
    potential = system.get_potential(config.shape, config.resolution)
    fig, _, _ = plot_potential_2d_x(potential)
    fig.show()

    hamiltonian = get_hamiltonian(system, config)
    eigenvectors = hamiltonian["basis"][0].vectors

    if index is None:
        for i in range(config.n_bands):
            state = get_state_vector(eigenvectors, i)
            fig, _ax, _line = plot_state_2d_x(state)
            fig.show()
            fig, _ax, _line = plot_state_2d_k(state)
            fig.show()
    else:
        state = get_state_vector(eigenvectors, index)
        fig, _ax, _line = plot_state_2d_x(state)
        fig.show()
        fig, _ax, _line = plot_state_2d_k(state)
        fig.show()

    input()


def plot_system_evolution_1d(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_B0],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
) -> None:
    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = ax.twinx()
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, ax, _anim = animate_state_over_list_1d_x(states, ax=ax1)

    fig.show()
    input()


def plot_system_evolution_2d(
    system: PeriodicSystem2d,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
) -> None:
    states = solve_schrodinger_equation(system, config, initial_state, times)

    fig, _ax, _anim = animate_state_over_list_2d_x(states)

    fig.show()
    input()


def plot_pair_system_evolution_1d(
    system: PeriodicSystem1d,
    config: PeriodicSystemConfig,
    initial_state: StateVector[_SBV0],
    times: EvenlySpacedTimeBasis[Any, Any, Any],
    direction: tuple[int] = (1,),
    *,
    measure: Measure = "abs",
) -> None:
    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, line = plot_potential_1d_x(potential)
    line.set_color("orange")
    ax1 = ax.twinx()

    (
        state_evolved_scattered,
        state_scattered_evolved,
    ) = get_isf_pair_states(system, config, initial_state, times, direction)

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

    fig, ax = plt.subplots()
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


def _get_default_times(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    direction: tuple[int, ...] = (1,),
) -> EvenlySpacedTimeBasis[Any, Any, Any]:
    return EvenlySpacedTimeBasis(
        100,
        1,
        -50,
        4 * get_free_particle_time(system, config, direction),
    )


def plot_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
    direction: tuple[int, ...] | None = None,
    *,
    n_repeats: int = 10,
    measure: Measure | None = None,
) -> None:
    direction = tuple(1 for _ in config.shape) if direction is None else direction

    times = _get_default_times(system, config, direction) if times is None else times
    data = get_boltzmann_isf(
        system,
        config,
        times,
        direction,
        n_repeats=n_repeats,
    )
    if measure is None:
        fig, ax, line = plot_value_list_against_time(data)
        line.set_label("abs ISF")

        fig, ax, line = plot_value_list_against_time(data, ax=ax, measure="real")
        line.set_label("real ISF")

        fig, ax, line = plot_value_list_against_time(data, ax=ax, measure="imag")
        line.set_label("imag ISF")
        ax.legend()
    else:
        fig, ax, line = plot_value_list_against_time(data, measure=measure)
    ax.set_ylabel("ISF")
    ax.set_title("Plot of the ISF against time")

    fig.show()

    fig, ax, line = plot_value_list_against_frequency(data)
    fig, ax, line = plot_value_list_against_frequency(data, measure="imag", ax=ax)
    fig, ax, line = plot_value_list_against_frequency(data, measure="real", ax=ax)
    ax.set_title("Plot of the fourier transform of the ISF against time")
    fig.show()

    fig.show()


def plot_band_resolved_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
    direction: tuple[int] = (1,),
    *,
    n_repeats: int = 10,
) -> None:
    times = _get_default_times(system, config, direction) if times is None else times

    resolved_data = get_band_resolved_boltzmann_isf(
        system,
        config,
        times,
        direction,
        n_repeats=n_repeats,
    )
    fig, _ = plot_split_value_list_against_time(resolved_data, measure="real")

    fig.show()

    fig, ax = plot_split_value_list_against_frequency(resolved_data)
    ax.set_title("Plot of the fourier transform of the ISF against time")
    fig.show()
    input()


def _calculate_effective_mass_from_gradient(
    temperature: float,
    gradient: float,
) -> float:
    return Boltzmann * temperature / (gradient**2)


@dataclass
class _AlphaDeltakFitData:
    """_Stores data from linear fit with calculated effective mass."""

    gradient: float
    intercept: float


def _get_alpha_deltak_linear_fit(
    values: ValueList[MomentumBasis],
) -> _AlphaDeltakFitData:
    k_points = values["basis"].k_points
    rates = values["data"]
    gradient, intercept = np.polyfit(k_points, rates, 1)
    return _AlphaDeltakFitData(gradient, intercept)


def _plot_alpha_deltak(
    data: ValueList[MomentumBasis],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    k_points = data["basis"].k_points

    fig, ax = get_figure(ax)

    ax.plot(k_points, data["data"], "bo", label="Bound")

    fit = _get_alpha_deltak_linear_fit(data)
    x_fit = np.array([0, k_points[len(k_points) - 1] * 1.2], dtype=np.float64)
    y_fit = fit.gradient * x_fit + fit.intercept
    ax.plot(x_fit, y_fit, "b")

    return fig, ax


def plot_alpha_deltak_comparison(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[tuple[int, ...]] | None = None,
    times: EvenlySpacedTimeBasis[Any, Any, Any] | None = None,
) -> None:
    data = get_ak_data(system, config, nk_points=nk_points, times=times)
    fig, ax = _plot_alpha_deltak(data)

    free_system = FreeSystem(system)
    free_data = get_ak_data(free_system, config, nk_points=nk_points, times=times)
    _, _ = _plot_alpha_deltak(free_data, ax=ax)

    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])

    print(  # noqa: T201
        "Bound mass =",
        _calculate_effective_mass_from_gradient(
            config.temperature,
            _get_alpha_deltak_linear_fit(data).gradient,
        ),
        "Free mass =",
        _calculate_effective_mass_from_gradient(
            config.temperature,
            _get_alpha_deltak_linear_fit(free_data).gradient,
        ),
    )

    fig.show()
    input()


def plot_scattered_energy_change_comparison(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    nk_points: list[tuple[int, ...]] | None = None,
) -> None:
    fig, ax = get_figure(None)

    bound_data = get_scattered_energy_change_against_k(
        system,
        config,
        nk_points=nk_points,
    )

    k_points = bound_data["basis"].k_points
    k_points_squared = np.square(k_points)

    ax.plot(k_points_squared, bound_data["data"], "blue", label="Bound")

    free_system = PeriodicSystem(
        id=system.id,
        barrier_energy=0,
        lattice_constant=system.lattice_constant,
        mass=system.mass,
    )
    free_data = get_scattered_energy_change_against_k(
        free_system,
        config,
        nk_points=nk_points,
        n_repeats=1,
    )
    ax.plot(k_points_squared, free_data["data"], "orange", label="Free")

    fig.legend()
    fig.show()
    input()
