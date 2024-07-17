from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.plot import plot_alpha_deltak
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), 50, temperature=155)
    system = SODIUM_COPPER_SYSTEM

    times = EvenlySpacedTimeBasis(100, 1, 0, 4e-13)
    k_points = [50, 75, 100, 125, 150, 175, 200]
    plot_alpha_deltak(system, config, nk_points=k_points)
