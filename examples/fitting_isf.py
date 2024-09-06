from __future__ import annotations

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import (
    DoubleGaussianMethod,
    ExponentialMethod,
    GaussianMethod,
    GaussianPlusExponentialMethod,
)
from coherent_rates.isf import (
    get_boltzmann_isf,
)
from coherent_rates.plot import plot_isf_with_fit
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((50,), (100,), direction=(10,), temperature=60)
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    times = GaussianPlusExponentialMethod("Gaussian").get_fit_times(
        system=system,
        config=config,
    )
    isf = get_boltzmann_isf(system, config, times, n_repeats=20)

    # Exponential fitting
    fig, ax = plot_isf_with_fit(
        isf,
        ExponentialMethod(),
        system=system,
        config=config,
    )
    ax.set_title("Exponential")  # type: ignore bad types
    fig.show()

    # Gaussian fitting
    fig, ax = plot_isf_with_fit(
        isf,
        GaussianMethod(),
        system=system,
        config=config,
    )
    ax.set_title("Gaussian")  # type: ignore bad types
    fig.show()

    # Gaussian + Exponential fitting
    fig, ax = plot_isf_with_fit(
        isf,
        GaussianPlusExponentialMethod("Gaussian"),
        system=system,
        config=config,
    )
    ax.set_title("Gaussian Plus Exponential")  # type: ignore bad types
    fig.show()

    # Double Gaussian fitting
    fig, ax = plot_isf_with_fit(
        isf,
        DoubleGaussianMethod("Fast"),
        system=system,
        config=config,
    )
    ax.set_title("Double Gaussian")  # type: ignore bad types
    fig.show()

    input()
