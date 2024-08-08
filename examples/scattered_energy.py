from surface_potential_analysis.state_vector.state_vector_list import get_state_vector

from coherent_rates.plot import (
    plot_occupation_against_energy_comparison,
    plot_scattered_energy_change_state,
    plot_thermal_scattered_energy_change_comparison,
)
from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM_1D,
    PeriodicSystemConfig,
    get_hamiltonian,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), 50, temperature=155)
    system = HYDROGEN_NICKEL_SYSTEM_1D

    # Shows that scattered energy is lower for a larger initial mass
    plot_occupation_against_energy_comparison(system, config, 3, (50,))

    hamiltonian = get_hamiltonian(system, config)
    n = 0
    b = 1
    k_points = [(n + b * i,) for i in range(10)]

    # For low state k, the dE vs dk plot is quadratic
    state = get_state_vector(hamiltonian["basis"][0].vectors, 0)
    plot_scattered_energy_change_state(system, config, state, nk_points=k_points)

    # For high state k, the dE vs dk plot is linear
    state = get_state_vector(hamiltonian["basis"][0].vectors, 230)
    plot_scattered_energy_change_state(system, config, state, nk_points=k_points)

    # Since dE is proportional to (k+dk)^2 - k^2 = 2k*dk +(dk)^2,
    # for low bands, k is small so dE~(dk)^2
    # for high bands, k is large so dE~dk

    # For a thermal state, we have <k> = 0 when averaging across experiments,
    # so the cross term averages out and we get dE proportional to (dk)^2
    plot_thermal_scattered_energy_change_comparison(system, config, nk_points=k_points)
