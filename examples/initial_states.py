from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_position_basis,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
)
from surface_potential_analysis.state_vector.plot import (
    plot_state_2d_k,
    plot_state_2d_x,
)
from surface_potential_analysis.util.plot import plot_data_2d_k, plot_data_2d_x

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.plot import plot_system_evolution_2d
from coherent_rates.state import (
    get_random_boltzmann_state,
    get_random_coherent_state,
    get_thermal_occupation_k,
    get_thermal_occupation_x,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((3, 3), (20, 20), temperature=155)
    system = SODIUM_COPPER_SYSTEM_2D
    times = EvenlySpacedTimeBasis(100, 1, 0, 3e-12)

    boltzmann_state = get_random_boltzmann_state(system, config)
    fig, ax, line = plot_state_2d_x(boltzmann_state)
    ax.set_title("Boltzmann state in real space")  # type: ignore unknown
    fig.show()
    fig, ax, line = plot_state_2d_k(boltzmann_state)
    ax.set_title("Boltzmann state in momentum space")  # type: ignore unknown
    fig.show()

    sigma = system.lattice_constant / 10

    coherent_state = get_random_coherent_state(system, config, sigma)
    fig, ax, line = plot_state_2d_x(coherent_state)
    ax.set_title("Coherent state in real space")  # type: ignore unknown
    fig.show()
    fig, ax, line = plot_state_2d_k(coherent_state)
    ax.set_title("Coherent state in momentum space")  # type: ignore unknown
    fig.show()

    plot_system_evolution_2d(system, config, coherent_state, times)

    potential = convert_potential_to_position_basis(
        system.get_potential(config.shape, config.resolution),
    )
    basis = potential["basis"]
    k_basis = stacked_basis_as_fundamental_momentum_basis(basis)

    x_probability_normalized = get_thermal_occupation_x(system, config)

    fig, ax, line = plot_data_2d_x(basis, x_probability_normalized)
    ax.set_title("probability distribution of initial position")  # type: ignore unknown
    fig.show()

    k_probability_normalized = get_thermal_occupation_k(system, config)
    fig, ax, line = plot_data_2d_k(k_basis, k_probability_normalized)
    ax.set_title("probability distribution of initial momentum")  # type: ignore unknown
    fig.show()
    input()
