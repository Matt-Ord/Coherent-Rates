from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((5,), (100,), 50)
    system = HYDROGEN_NICKEL_SYSTEM
