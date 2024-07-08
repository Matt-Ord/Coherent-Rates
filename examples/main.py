from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.plot import plot_boltzmann_isf
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM,
    PeriodicSystemConfig,
    get_extended_interpolated_potential,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), 50)
    system = HYDROGEN_NICKEL_SYSTEM
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )

    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-13)
    plot_boltzmann_isf(system, config, 155, times, (54,))
