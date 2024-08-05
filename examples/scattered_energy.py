from coherent_rates.plot import (
    plot_scattered_energy_change_comparison,
)
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystemConfig,
)

# Generates a plot of energy change due to scattering against square of scattered
# momentum for both free and bound particles

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), 50, temperature=155)
    system = HYDROGEN_NICKEL_SYSTEM_1D

    n = 0
    b = 1
    k_points = [(n + b * i,) for i in range(10)]

    plot_scattered_energy_change_comparison(system, config, nk_points=k_points)
