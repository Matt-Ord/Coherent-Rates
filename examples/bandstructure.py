from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import ExponentialMethod
from coherent_rates.plot import plot_boltzmann_isf, plot_system_bands
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    plot_system_bands(system, config)

    times = ExponentialMethod().get_fit_times(system=system, config=config)
    plot_boltzmann_isf(system, config, times)
