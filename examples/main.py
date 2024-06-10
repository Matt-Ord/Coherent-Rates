from coherent_rates.plot import plot_system_eigenstates, plot_system_evolution
from coherent_rates.system import HYDROGEN_NICKEL_SYSTEM, PeriodicSystemConfig

if __name__ == "__main__":
    config = PeriodicSystemConfig((1,), (100,), 10)

    plot_system_eigenstates(HYDROGEN_NICKEL_SYSTEM, config)

    plot_system_evolution(HYDROGEN_NICKEL_SYSTEM, config)
