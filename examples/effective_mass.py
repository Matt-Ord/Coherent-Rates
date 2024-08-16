from coherent_rates.plot import plot_ak_temp_data
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((10, 1), (15, 15), 225, temperature=155)
    system = SODIUM_COPPER_SYSTEM_2D

    nk_points = [(2 * i + 5, 0) for i in range(5)]
    temps = [(60 + 30 * i) for i in range(5)]

    plot_ak_temp_data(system, config, temperatures=temps, nk_points=nk_points)
