from coherent_rates.fit import GaussianMethod
from coherent_rates.plot import (
    plot_rate_against_mass_and_momentum_data,
    plot_rate_against_temperature_and_momentum_data,
)
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((50,), (100,), temperature=155)
    system = HYDROGEN_NICKEL_SYSTEM_1D

    nk_points = [(50,), (75,), (100,), (125,), (150,), (175,), (200,)]
    temperatures = [(50.0 + 100 * i) for i in range(10)]
    masses = [(1 + 1000 * i) * system.mass for i in range(5)]

    plot_rate_against_mass_and_momentum_data(
        system,
        config,
        masses=masses,
        nk_points=nk_points,
        fit_method=GaussianMethod(),
    )

    plot_rate_against_temperature_and_momentum_data(
        system,
        config,
        temperatures=temperatures,
        nk_points=nk_points,
        fit_method=GaussianMethod(),
    )
