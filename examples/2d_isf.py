from surface_potential_analysis.potential.plot import plot_potential_2d_x

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import GaussianMethod
from coherent_rates.plot import (
    plot_boltzmann_isf,
    plot_system_bands,
    plot_system_eigenstates_2d,
    plot_system_eigenvalues,
    plot_system_evolution_2d,
)
from coherent_rates.solve import get_bloch_wavefunctions
from coherent_rates.state import get_random_boltzmann_state
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    SODIUM_COPPER_SYSTEM_2D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (20, 20),
        (30, 30),
        truncation=150,
        direction=(3, 0),
        temperature=155,
    )
    system = SODIUM_COPPER_SYSTEM_2D

    potential = system.get_potential(config.shape, config.resolution)
    fig, ax, _ = plot_potential_2d_x(potential)
    fig.show()
    input()

    get_bloch_wavefunctions.load_or_call_cached(system, config)

    plot_system_eigenstates_2d(system, config, states=[0, 149])
    plot_system_bands(system, config)
    plot_system_eigenvalues(system, config)

    times = GaussianMethod().get_fit_times(
        system=system,
        config=config,
    )
    state = get_random_boltzmann_state(system, config)
    plot_system_evolution_2d(system, config, state, times)

    plot_boltzmann_isf(system, config, times, n_repeats=10)

    # The equivalent ISF in 1d: Note
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=30,
        direction=(3 * 5,),
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    plot_boltzmann_isf(system, config, times, n_repeats=10)
