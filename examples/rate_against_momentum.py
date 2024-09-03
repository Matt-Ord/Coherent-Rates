from __future__ import annotations

from typing import TYPE_CHECKING, Any

from surface_potential_analysis.util.plot import get_figure

from coherent_rates.fit import GaussianMethod
from coherent_rates.plot import (
    plot_rate_against_momentum,
    plot_rate_against_momentum_isf_fit,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    FreeSystem,
    PeriodicSystem,
    PeriodicSystemConfig,
)

if TYPE_CHECKING:
    from coherent_rates.fit import FitMethod


def _test_convergence_with_shape(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    directions: list[tuple[int, ...]],
    fit_method: FitMethod[Any] | None = None,
) -> None:
    fig, ax, line = plot_rate_against_momentum(
        system,
        config,
        directions=directions,
        fit_method=fit_method,
    )
    line.set_label("Standard")
    directions = [tuple(2 * j for j in i) for i in directions]
    _, _, line = plot_rate_against_momentum(
        system,
        config.with_shape(tuple(2 * j for j in config.shape)),
        directions=directions,
        fit_method=fit_method,
        ax=ax,
    )
    line.set_label("2x shape")
    ax.legend()  # type: ignore unknown
    fig.show()
    input()


def _compare_rate_against_free_surface(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> None:
    fit_method = GaussianMethod() if fit_method is None else fit_method

    fig, ax = get_figure(None)

    _, _, line = plot_rate_against_momentum(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label(f"Bound system, {fit_method.get_rate_label()}")

    _, _, line = plot_rate_against_momentum(
        FreeSystem(system),
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label(f"Free system, {fit_method.get_rate_label()}")

    ax.legend()  # type: ignore library type
    ax.set_title("Plot of rate against delta k, comparing to a free particle")  # type: ignore library type

    fig.show()
    input()


if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=50,
        temperature=100,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D
    directions = [(i,) for i in [1, 2, *list(range(5, 105, 5))]]

    _test_convergence_with_shape(system, config, directions=directions)
    plot_rate_against_momentum_isf_fit(system, config, directions=directions)
    _compare_rate_against_free_surface(system, config, directions=directions)
