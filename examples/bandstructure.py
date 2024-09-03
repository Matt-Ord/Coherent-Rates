from coherent_rates.fit import ExponentialMethod
from coherent_rates.plot import plot_boltzmann_isf, plot_system_bands
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((80,), (50,), 8, temperature=400)
    system = HYDROGEN_NICKEL_SYSTEM_1D

    plot_system_bands(system, config)
    times = ExponentialMethod().get_fit_times(system=system, config=config)
    plot_boltzmann_isf(system, config, times)
