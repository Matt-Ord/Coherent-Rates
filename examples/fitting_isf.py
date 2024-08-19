from typing import Any, TypeVar

import numpy as np
from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.state_vector.eigenstate_list import ValueList
from surface_potential_analysis.state_vector.plot_value_list import (
    plot_value_list_against_time,
)

from coherent_rates.fit import (
    DoubleGaussianMethod,
    ExponentialMethod,
    FitMethod,
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

_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])


def plot_isf_with_fit(
    data: ValueList[_BT0],
    method: FitMethod[Any],
) -> None:
    fig, ax, line = plot_value_list_against_time(data)
    line.set_label("ISF")

    fit = method.get_fit_from_isf(data)
    fitted_data = method.get_fitted_data(fit, data["basis"])

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax)
    line.set_label("Fit")

    ax.legend()  # type: ignore bad types
    fig.show()
    input()


if __name__ == "__main__":
    config = PeriodicSystemConfig((10, 1), (15, 15), 225, temperature=60)
    system = SODIUM_COPPER_SYSTEM_2D

    n = 7
    direction = (n, 0)
    times = EvenlySpacedTimeBasis(200, 1, 0, 2e-11)
    potential = system.get_potential(config.shape, config.resolution)
    dk_stacked = BasisUtil(potential["basis"]).dk_stacked
    k_length = np.linalg.norm(np.einsum("j,jk->k", direction, dk_stacked))  # type: ignore bad types

    isf = get_boltzmann_isf(system, config, times, direction, n_repeats=20)

    # Exponential fitting
    exp = ExponentialMethod().get_fit_from_isf(isf)
    print(exp)
    print(k_length)
    print(ExponentialMethod.get_rates_from_fit(exp))
    plot_isf_with_fit(isf, ExponentialMethod())

    # Gaussian fitting
    gauss = GaussianMethod().get_fit_from_isf(isf)
    print(gauss)
    print(k_length)
    print(GaussianMethod.get_rates_from_fit(gauss))
    plot_isf_with_fit(isf, GaussianMethod())

    # Guassian + Exponential fitting
    gauss, exp = GaussianPlusExponentialMethod().get_fit_from_isf(isf)
    print(gauss, exp)
    print(k_length)
    print(GaussianPlusExponentialMethod.get_rates_from_fit((gauss, exp)))
    plot_isf_with_fit(isf, GaussianPlusExponentialMethod())

    # Double Gaussian fitting
    gauss1, gauss2 = DoubleGaussianMethod().get_fit_from_isf(isf)
    print(gauss1, gauss2)
    print(k_length)
    print(DoubleGaussianMethod.get_rates_from_fit((gauss1, gauss2)))
    plot_isf_with_fit(isf, DoubleGaussianMethod())
