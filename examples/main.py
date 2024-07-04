from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.system import (
    HYDROGEN_NICKEL_SYSTEM,
    PeriodicSystemConfig,
    get_extended_interpolated_potential,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((2,), (6,), 6)
    system = HYDROGEN_NICKEL_SYSTEM
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    times = EvenlySpacedTimeBasis(100, 1, 0, 1e-13)
    # hamiltonian = get_hamiltonian(system, config)
    # oper = get_cl_test(system, config, 300)
    # data = np.log(get_measured_data(oper["data"], "real"))
    # # plt.imshow(data, interpolation="none")
    # # plt.show()

    # opercon = convert_operator_to_basis(oper, hamiltonian["basis"])
    # eigmax = opercon["data"].reshape(
    #     (int(np.sqrt(len(opercon["data"]))), int(np.sqrt(len(opercon["data"])))),
    # )

    # eigval, eigvec = np.linalg.eig(eigmax)
    # eigval = eigval / sum(eigval)
    # print(eigval)
    # print(get_cl_stationary_states_1d(system, config, 300)["eigenvalue"])
    get_aver
