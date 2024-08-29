from coherent_rates.fit import GaussianMethod
from coherent_rates.plot import (
    plot_rate_against_mass_and_momentum,
    plot_rate_against_temperature_and_momentum,
    plot_rate_against_temperature_barrier_energy_and_momentum,
    plot_rate_against_temperature_mass_and_momentum,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((50,), (100,), temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    nk_points = [(50,), (75,), (100,), (125,), (150,), (175,), (200,)]
    temperatures = [(10.0 + 50 * i) for i in range(5)]
    r = [0.1, 0.5, 1, 5, 10]
    masses = [i * system.mass for i in r]
    barrier_energies = [i * system.barrier_energy for i in r]

    plot_rate_against_mass_and_momentum(
        system,
        config,
        masses=masses,
        nk_points=nk_points,
        fit_method=GaussianMethod(),
    )

    plot_rate_against_temperature_and_momentum(
        system,
        config,
        temperatures=temperatures,
        nk_points=nk_points,
        fit_method=GaussianMethod(),
    )

    plot_rate_against_temperature_mass_and_momentum(
        system,
        config,
        temperatures=temperatures,
        masses=masses,
        nk_points=nk_points,
        fit_method=GaussianMethod(),
    )

    plot_rate_against_temperature_barrier_energy_and_momentum(
        system,
        config,
        temperatures=temperatures,
        barrier_energies=barrier_energies,
        nk_points=nk_points,
        fit_method=GaussianMethod(),
    )
