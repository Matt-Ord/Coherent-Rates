from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from surface_potential_analysis.basis.time_basis_like import (
    BasisWithTimeLike,
    EvenlySpacedTimeBasis,
)
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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from surface_potential_analysis.state_vector.eigenstate_list import ValueList

_BT0 = TypeVar("_BT0", bound=BasisWithTimeLike[Any, Any])


def plot_isf_with_fit(
    data: ValueList[_BT0],
    method: FitMethod[Any],
) -> tuple[Figure, Axes]:
    fig, ax, line = plot_value_list_against_time(data)
    line.set_label("ISF")

    fit = method.get_fit_from_isf(data)
    fitted_data = method.get_fitted_data(fit, data["basis"])

    fig, ax, line = plot_value_list_against_time(fitted_data, ax=ax)
    line.set_label("Fit")

    ax.legend()  # type: ignore bad types
    return (fig, ax)


print(GaussianMethod.__repr__(GaussianMethod()))
if __name__ == "__main__":
    config = PeriodicSystemConfig((10, 1), (15, 15), 225, temperature=60)
    system = SODIUM_COPPER_SYSTEM_2D

    n = 7
    direction = (n, 0)
    times = EvenlySpacedTimeBasis(200, 1, 0, 2e-11)
    isf = get_boltzmann_isf(system, config, times, direction, n_repeats=20)

    # Exponential fitting
    fig, ax = plot_isf_with_fit(isf, ExponentialMethod())
    ax.set_title("Exponential")  # type: ignore bad types
    fig.show()

    # Gaussian fitting
    fig, ax = plot_isf_with_fit(isf, GaussianMethod())
    ax.set_title("Gaussian")  # type: ignore bad types
    fig.show()

    # Gaussian + Exponential fitting
    fig, ax = plot_isf_with_fit(isf, GaussianPlusExponentialMethod())
    ax.set_title("Gaussian Plus Exponential")  # type: ignore bad types
    fig.show()

    # Double Gaussian fitting
    fig, ax = plot_isf_with_fit(isf, DoubleGaussianMethod())
    ax.set_title("Double Gaussian")  # type: ignore bad types
    fig.show()

    input()
