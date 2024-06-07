from coherent_rates.plot import plot_system_eigenstates
from coherent_rates.system import HYDROGEN_NICKEL_SYSTEM, PeriodicSystemConfig

if __name__ == "__main__":
    config = PeriodicSystemConfig((1,), (100,), 2)

    plot_system_eigenstates(HYDROGEN_NICKEL_SYSTEM, config)
