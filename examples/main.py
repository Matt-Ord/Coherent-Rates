from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.plot import (
    plot_band_resolved_boltzmann_isf,
    plot_boltzmann_isf,
    plot_pair_system_evolution_1d,
)
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((5,), (50,), 50, temperature=150)
    system = HYDROGEN_NICKEL_SYSTEM_1D

    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-13)

    plot_pair_system_evolution_1d(system, config, times, direction=(10,))
    plot_boltzmann_isf(system, config, times, (10,))
    plot_band_resolved_boltzmann_isf(system, config, times, (1,), n_repeats=10)
    plot_band_resolved_boltzmann_isf(system, config, times, (1,), n_repeats=100)
