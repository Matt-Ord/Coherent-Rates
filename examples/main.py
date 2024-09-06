from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.plot import (
    plot_band_resolved_boltzmann_isf,
    plot_boltzmann_isf,
    plot_free_isf,
    plot_pair_system_evolution_1d,
    plot_system_bands,
    plot_system_eigenstates_1d,
    plot_system_eigenvalues,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((50,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-13)

    plot_system_eigenstates_1d(system, config)
    plot_system_eigenvalues(system, config)
    plot_system_bands(system, config)
    plot_pair_system_evolution_1d(system, config, times)
    plot_free_isf(system, config, times)
    plot_boltzmann_isf(system, config, times)
    plot_band_resolved_boltzmann_isf(system, config, times, n_repeats=100)
