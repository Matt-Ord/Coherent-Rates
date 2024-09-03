import numpy as np

from coherent_rates.isf import get_conditions_at_barrier_energy, get_conditions_at_mass
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
    config = PeriodicSystemConfig((100,), (100,), truncation=50, temperature=155)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    directions = [(i,) for i in [1, 2, *list(range(5, 105, 5))]]
    masses = np.array([0.1, 0.5, 1, 5, 10]) * system.mass
    plot_effective_mass_against_mass(
        system,
        config,
        masses=masses,
        directions=directions,
    )

    temperatures = np.array([(20 * i) for i in range(3, 10)])
    (fig, ax, line), momentum_fig = plot_effective_mass_against_temperature(
        system,
        config,
        temperatures=temperatures,
        directions=directions,
    )
    plot_barrier_temperature(system.barrier_energy, ax=ax)
    fig.show()
    momentum_fig[0].show()
    input()

    conditions = get_conditions_at_mass(system, config, masses)
    plot_effective_mass_against_temperature_comparison(
        conditions,
        temperatures=temperatures,
        directions=directions,
    )

    barrier_energies = np.array([0.1, 0.5, 1, 5, 10]) * system.barrier_energy
    conditions = get_conditions_at_barrier_energy(system, config, barrier_energies)
    plot_effective_mass_against_temperature_comparison(
        conditions,
        temperatures=temperatures,
        directions=directions,
    )
