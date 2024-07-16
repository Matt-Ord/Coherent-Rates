from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.plot import plot_boltzmann_isf, plot_system_bands
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), 8, temperature=155)
    system = HYDROGEN_NICKEL_SYSTEM

    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-13)

    plot_system_bands(system, config)
    plot_boltzmann_isf(system, config, times, (54,))
