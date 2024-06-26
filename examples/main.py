import numpy as np
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_position_basis,
)

from coherent_rates.plot import plot_pair_system_evolution
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM,
    PeriodicSystemConfig,
    get_extended_interpolated_potential,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((5,), (100,), 50)
    system = HYDROGEN_NICKEL_SYSTEM
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )

    initial_state = {
        "basis": stacked_basis_as_fundamental_position_basis(potential["basis"]),
        "data": np.zeros(500, dtype=np.complex128),
    }
    for i in range(50):
        initial_state["data"][i] = 1

    times0 = EvenlySpacedTimeBasis(100, 1, 0, 5e-14)
    plot_pair_system_evolution(system, config, initial_state, times0)
