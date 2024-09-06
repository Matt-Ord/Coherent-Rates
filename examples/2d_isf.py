from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.potential.plot import plot_potential_2d_x

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.plot import (
    plot_boltzmann_isf,
    plot_system_bands,
    plot_system_eigenstates_2d,
    plot_system_eigenvalues,
    plot_system_evolution_2d,
)
from coherent_rates.state import get_random_boltzmann_state
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (20, 20),
        (30, 30),
        truncation=100,
        direction=(1, 0),
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, _ = plot_potential_2d_x(potential)
    fig.show()
    input()

    plot_system_eigenstates_2d(system, config, states=[0])
    plot_system_eigenvalues(system, config)
    plot_system_bands(system, config)

    times = EvenlySpacedTimeBasis(101, 1, -50, 1e-11)
    state = get_random_boltzmann_state(system, config)
    plot_system_evolution_2d(system, config, state, times)

    times = EvenlySpacedTimeBasis(101, 1, -50, 0.2e-11)
    isf = plot_boltzmann_isf(system, config, times, n_repeats=10)
