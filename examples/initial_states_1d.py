from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot import (
    plot_state_1d_k,
    plot_state_1d_x,
)
from surface_potential_analysis.util.plot import (
    plot_data_1d_k,
    plot_data_1d_x,
)

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.plot import plot_system_evolution_1d
from coherent_rates.state import (
    get_random_boltzmann_state,
    get_random_coherent_k,
    get_random_coherent_state,
    get_random_coherent_x,
    get_thermal_occupation_k,
    get_thermal_occupation_x,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((3,), (30,), temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    times = EvenlySpacedTimeBasis(100, 1, 0, 3e-12)

    boltzmann_state = get_random_boltzmann_state(system, config)
    fig, ax, line = plot_state_1d_x(boltzmann_state)
    ax.set_title("Boltzmann state in real space")  # type: ignore unknown
    fig.show()
    fig, ax, line = plot_state_1d_k(boltzmann_state)
    ax.set_title("Boltzmann state in momentum space")  # type: ignore unknown
    fig.show()

    sigma = system.lattice_constant / 10

    coherent_state = get_random_coherent_state(system, config, (sigma,))
    fig, ax, line = plot_state_1d_x(coherent_state)
    ax.set_title("Coherent state in real space")  # type: ignore unknown
    fig.show()
    fig, ax, line = plot_state_1d_k(coherent_state)
    ax.set_title("Coherent state in momentum space")  # type: ignore unknown
    fig.show()
    input()
    plot_system_evolution_1d(system, config, coherent_state, times)

    # Check the x, p distribution used in the coherent states
    config = config.with_shape((1,))
    x_coords = [get_random_coherent_x(system, config)[0] for _ in range(10000)]
    x_probability_normalized = get_thermal_occupation_x(system, config)

    fig, ax, line = plot_data_1d_x(
        x_probability_normalized["basis"],
        x_probability_normalized["data"],
    )
    ax.twinx().hist(x_coords, density=False, bins=20)  # type: ignore unknown
    ax.set_title("probability distribution of initial position")  # type: ignore unknown
    fig.show()

    k_coords = [get_random_coherent_k(system, config)[0] for _ in range(10000)]
    k_probability_normalized = get_thermal_occupation_k(system, config)
    fig, ax, line = plot_data_1d_k(
        k_probability_normalized["basis"],
        k_probability_normalized["data"].reshape(
            k_probability_normalized["basis"].shape,
        ),
    )
    ax.twinx().hist(k_coords, density=False, bins=20)  # type: ignore unknown
    ax.set_title("probability distribution of initial momentum")  # type: ignore unknown
    fig.show()
    input()
