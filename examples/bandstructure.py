from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.isf import get_free_particle_time
from coherent_rates.plot import plot_boltzmann_isf, plot_system_bands
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((80,), (50,), 8, temperature=400)
    system = HYDROGEN_NICKEL_SYSTEM_1D

    plot_system_bands(system, config)
    times = EvenlySpacedTimeBasis(
        100,
        1,
        0,
        20 * get_free_particle_time(system, config, (1,)),
    )
    plot_boltzmann_isf(system, config, times, direction=(1,))
