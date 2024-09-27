from __future__ import annotations

from typing import TYPE_CHECKING, Any

from surface_potential_analysis.util.plot import get_figure

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    GaussianMethod,
)
from coherent_rates.plot import (
    plot_rate_against_momentum,
    plot_rate_against_momentum_isf_fit,
)
from coherent_rates.solve import get_hamiltonian
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM_2D,
    FreeSystem,
    System,
)

if TYPE_CHECKING:
    from coherent_rates.fit import FitMethod


def _compare_rate_against_free_surface(
    system: System,
    config: PeriodicSystemConfig,
    *,
    fit_method: FitMethod[Any] | None = None,
    free_fit_method: FitMethod[Any] | None = None,
    directions: list[tuple[int, ...]] | None = None,
) -> None:
    fit_method = GaussianMethod() if fit_method is None else fit_method
    free_fit_method = GaussianMethod() if free_fit_method is None else free_fit_method

    fig, ax = get_figure(None)

    _, _, line = plot_rate_against_momentum(
        system,
        config,
        fit_method=fit_method,
        directions=directions,
        ax=ax,
    )
    line.set_label(f"Bound system, {fit_method.get_rate_label()}")

    get_hamiltonian.load_or_call_cached(FreeSystem(system), config)
    _, _, line = plot_rate_against_momentum(
        FreeSystem(system),
        config,
        fit_method=free_fit_method,
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
        (20, 20),
        (30, 30),
        truncation=150,
        temperature=100,
    )
    system = SODIUM_COPPER_SYSTEM_2D
    system = FreeSystem(system)
    directions = [(i, 0) for i in [1, 2, *list(range(5, 55, 5))]]

    get_hamiltonian.load_or_call_cached(system, config)
    plot_rate_against_momentum_isf_fit(
        system,
        config,
        directions=directions,
        fit_method=GaussianMethod(truncate=False),
    )
    _compare_rate_against_free_surface(
        system,
        config,
        directions=directions,
        fit_method=GaussianMethod(truncate=False),
        free_fit_method=GaussianMethod(truncate=False),
    )
