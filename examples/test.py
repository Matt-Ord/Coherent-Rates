import numpy as np
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)
from surface_potential_analysis.util.plot import plot_data_1d

from coherent_rates.isf import (
    fit_abs_isf_to_exponential,
    fit_abs_isf_to_exponential_and_gaussian,
    fit_abs_isf_to_gaussian_constant,
    get_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((10, 1), (15, 15), 225, temperature=155)
    system = SODIUM_COPPER_SYSTEM_2D
    times = EvenlySpacedTimeBasis(200, 1, 0, 10e-12)
    n = 1
    direction = (n, 0)

    potential = system.get_potential(config.shape, config.resolution)
    dk_stacked = BasisUtil(potential["basis"]).dk_stacked
    k_length = np.linalg.norm(np.einsum("j,jk->k", direction, dk_stacked))

    isf = get_boltzmann_isf(system, config, times, direction, n_repeats=20)

    exp = fit_abs_isf_to_exponential(isf)
    print(exp)
    fig, ax, line = plot_value_list_against_time(isf)
    line.set_label("isf")

    time = times.times
    y_fit = (1 - exp.amplitude) + exp.amplitude * np.exp(-1 * time / exp.time_constant)
    fig, ax, line = plot_data_1d(y_fit, time, ax=ax, measure="real")
    line.set_label("fit")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.show()
    print(k_length)
    print(exp.time_constant)
    print(exp.time_constant_error)

    gauss = fit_abs_isf_to_gaussian_constant(isf)
    print(gauss)

    fig, ax, line = plot_value_list_against_time(isf)
    line.set_label("isf")

    time = times.times
    y_fit = gauss.constant + gauss.amplitude * np.exp(
        -1 * np.square(time / gauss.width) / 2,
    )
    fig, ax, line = plot_data_1d(y_fit, time, ax=ax, measure="real")
    line.set_label("fit")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.show()
    print(k_length)
    print(gauss.width)
    print(gauss.width_error)
    input()

    exp, gauss = fit_abs_isf_to_exponential_and_gaussian(isf)
    print(exp, gauss)

    fig, ax, line = plot_value_list_against_time(isf)
    line.set_label("isf")

    time = times.times
    y_fit = (
        (1 - exp.amplitude - gauss.amplitude)
        + exp.amplitude * np.exp(-1 * time / exp.time_constant)
        + gauss.amplitude * np.exp(-1 * np.square(time / gauss.width) / 2)
    )
    gauss_fit = (1 - exp.amplitude - gauss.amplitude) + gauss.amplitude * np.exp(
        -1 * np.square(time / gauss.width) / 2,
    )
    fig, ax, line = plot_data_1d(gauss_fit, time, ax=ax, measure="real")
    line.set_label("gauss fit")
    fig, ax, line = plot_data_1d(y_fit, time, ax=ax, measure="real")
    ax.set_ylim(0, 1)
    line.set_label("fit")
    ax.legend()
    fig.show()

    print("amps:", exp.amplitude, gauss.amplitude)
    print(k_length)
    print(exp.time_constant)
    print(exp.time_constant_error)
    print(gauss.width)
    print(gauss.width_error)
    input()
