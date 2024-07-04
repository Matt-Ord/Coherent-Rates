import numpy as np
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.operator.conversion import convert_operator_to_basis
from surface_potential_analysis.util.plot import plot_data_2d

from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM,
    PeriodicSystemConfig,
    get_cl_operator,
    get_extended_interpolated_potential,
    get_hamiltonian,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((5,), (7,), 3)
    system = HYDROGEN_NICKEL_SYSTEM
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-13)
    hamiltonian = get_hamiltonian(system, config)

    operator = get_cl_operator(system, config, 155)
    fig, ax, line = plot_data_2d(operator["data"])
    fig.show()
    operator_converted = convert_operator_to_basis(operator, hamiltonian["basis"])
    eigmax = operator_converted["data"].reshape(
        (hamiltonian["basis"].shape),
    )
    eigval, eigvec = np.linalg.eig(eigmax)
    eigval = eigval / sum(eigval)
    print(eigval)
    input()
