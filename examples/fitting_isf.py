import numpy as np
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.basis.util import BasisUtil

from coherent_rates.fit import (
    DoubleGaussianMethod,
    ExponentialMethod,
    GaussianMethod,
    GaussianPlusExponentialMethod,
)
from coherent_rates.isf import (
    get_boltzmann_isf,
)
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((10, 1), (15, 15), 225, temperature=60)
    system = SODIUM_COPPER_SYSTEM_2D

    n = 7
    direction = (n, 0)
    times = EvenlySpacedTimeBasis(200, 1, 0, 2e-11)
    potential = system.get_potential(config.shape, config.resolution)
    dk_stacked = BasisUtil(potential["basis"]).dk_stacked
    k_length = np.linalg.norm(np.einsum("j,jk->k", direction, dk_stacked))

    isf = get_boltzmann_isf(system, config, times, direction, n_repeats=20)

    # Exponential fitting
    exp = ExponentialMethod().fit_isf_data(isf)
    print(exp)
    print(k_length)
    print(ExponentialMethod().get_rates_from_fit(exp))
    ExponentialMethod().plot_isf_with_fit(isf)

    # Gaussian fitting
    gauss = GaussianMethod().fit_isf_data(isf)
    print(gauss)
    print(k_length)
    print(GaussianMethod().get_rates_from_fit(gauss))
    GaussianMethod().plot_isf_with_fit(isf)

    # Guassian + Exponential fitting
    gauss, exp = GaussianPlusExponentialMethod().fit_isf_data(isf)
    print(gauss, exp)
    print(k_length)
    print(GaussianPlusExponentialMethod().get_rates_from_fit((gauss, exp)))
    GaussianPlusExponentialMethod().plot_isf_with_fit(isf)

    # Double Gaussian fitting
    gauss1, gauss2 = DoubleGaussianMethod().fit_isf_data(isf)
    print(gauss1, gauss2)
    print(k_length)
    print(DoubleGaussianMethod().get_rates_from_fit((gauss1, gauss2)))
    DoubleGaussianMethod().plot_isf_with_fit(isf)
