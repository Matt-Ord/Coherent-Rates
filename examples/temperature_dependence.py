import numpy as np

from coherent_rates.plot import (
    plot_barrier_temperature,
    plot_effective_mass_against_mass,
    plot_effective_mass_against_temperature,
    plot_effective_mass_against_temperature_comparison,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((50,), (100,), temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    nk_points = [(50,), (75,), (100,), (125,), (150,), (175,), (200,)]

    masses = np.array([0.1, 0.5, 1, 5, 10]) * system.mass
    plot_effective_mass_against_mass(
        system,
        config,
        masses=masses,
        nk_points=nk_points,
    )

    temperatures = np.array([(10.0 + 50 * i) for i in range(5)])
    (fig, ax, line), _ = plot_effective_mass_against_temperature(
        system,
        config,
        temperatures=temperatures,
        nk_points=nk_points,
    )
    plot_barrier_temperature(system.barrier_energy, ax=ax)
    fig.show()
    input()

    systems = [(system.with_mass(mass), f"{mass}") for mass in masses]
    plot_effective_mass_against_temperature_comparison(
        systems,
        config,
        temperatures=temperatures,
        nk_points=nk_points,
    )

    barrier_energies = np.array([0.1, 0.5, 1, 5, 10]) * system.barrier_energy
    systems = [
        (system.with_barrier_energy(barrier_energy), f"{barrier_energy}")
        for barrier_energy in barrier_energies
    ]
    plot_effective_mass_against_temperature_comparison(
        systems,
        config,
        temperatures=temperatures,
        nk_points=nk_points,
    )
