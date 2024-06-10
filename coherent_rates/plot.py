import numpy as np
from matplotlib import pyplot as plt
from surface_potential_analysis.basis.explicit_basis import (
    explicit_stacked_basis_as_fundamental,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.potential.plot import plot_potential_1d_x
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.conversion import (
    convert_state_vector_list_to_basis,
)
from surface_potential_analysis.state_vector.plot import (
    animate_state_over_list_1d_x,
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
    basis = explicit_stacked_basis_as_fundamental(hamiltonian["basis"][0])
    converted = convert_state_vector_list_to_basis(eigenvectors, basis)

    ax1 = ax.twinx()
    fig2, ax2 = plt.subplots()
    for _i, state in enumerate(state_vector_list_into_iter(converted)):
        plot_state_1d_x(state, ax=ax1)

        plot_state_1d_k(state, ax=ax2)

    fig.show()
    fig2.show()
    input()


def plot_system_evolution(
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

    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-14)
    initial_state = {
        "basis": stacked_basis_as_fundamental_position_basis(potential["basis"]),
        "data": np.zeros(100, dtype=np.complex128),
    }
    initial_state["data"][0] = 1

    states = solve_schrodinger_equation(system, config, initial_state, times)

    basis = explicit_stacked_basis_as_fundamental(states["basis"][1])
    converted = convert_state_vector_list_to_basis(states, basis)

    fig, ax, _anim = animate_state_over_list_1d_x(converted)

    fig, ax, _anim0 = animate_state_over_list_1d_x(converted, ax=ax, measure="real")
    fig.show()
    input()
