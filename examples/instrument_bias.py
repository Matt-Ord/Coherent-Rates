import numpy as np
from scipy.constants import electron_volt
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_frequency,
    plot_value_list_against_time,
)
from surface_potential_analysis.util.plot import get_figure

from coherent_rates.isf import (
    get_boltzmann_isf,
    get_conditions_at_energy_range,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((50,), (100,), 50, temperature=150)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    times = EvenlySpacedTimeBasis(201, 1, -100, 5e-11)
    direction = (10,)
    n_repeats = 20

    fig0, ax0 = get_figure(None)
    fig1, ax1 = get_figure(None)
    energies = np.arange(0.5, 2.5, 0.5) * 0.001 * electron_volt
    for s, c, label in get_conditions_at_energy_range(
        system,
        config,
        energies,
    ):
        data = get_boltzmann_isf(
            s,
            c,
            times,
            direction,
            n_repeats=n_repeats,
        )

        _, _, line = plot_value_list_against_time(data, ax=ax0)
        line.set_label(label)
        _, _, line = plot_value_list_against_frequency(data, ax=ax1)
        line.set_label(label)

    ax0.legend()
    fig0.show()
    ax1.legend()
    fig1.show()
    input()
