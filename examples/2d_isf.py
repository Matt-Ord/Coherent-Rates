from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.isf import get_random_boltzmann_state
from coherent_rates.plot import (
    plot_alpha_deltak_comparison,
    plot_boltzmann_isf,
    plot_system_eigenstates_2d,
    plot_system_evolution_2d,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((5, 5), (10, 10), 100, temperature=155)
    system = SODIUM_COPPER_SYSTEM_2D

    plot_system_eigenstates_2d(system, config, 0)

    times = EvenlySpacedTimeBasis(101, 1, -50, 1e-11)
    state = get_random_boltzmann_state(system, config)
    plot_system_evolution_2d(system, config, state, times)

    direction = (5, 0)
    isf = plot_boltzmann_isf(system, config, times, direction, n_repeats=10)

    nk_points = [(0, 3 * i) for i in range(1, 5)]
    plot_alpha_deltak_comparison(system, config, nk_points=nk_points)
