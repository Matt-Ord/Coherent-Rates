from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.constants import electron_volt  # type: ignore bad library

from coherent_rates.config import PeriodicSystemConfig
from coherent_rates.fit import FitMethod, GaussianMethodWithOffset
from coherent_rates.isf import (
    get_conditions_at_energy_range,
)
from coherent_rates.plot import (
    plot_boltzmann_isf_fit_for_conditions,
)
from coherent_rates.system import (
    SODIUM_COPPER_BRIDGE_SYSTEM_1D,
    System,
)


def _test_instrument_bias(
    system: System,
    config: PeriodicSystemConfig,
    energies: Iterable[float],
    *,
    fit_method: FitMethod[Any] | None = None,
) -> None:
    fit_method = GaussianMethodWithOffset() if fit_method is None else fit_method
    conditions = get_conditions_at_energy_range(
        system,
        config,
        energies,
    )
    plot_boltzmann_isf_fit_for_conditions(conditions, fit_method=fit_method)


if __name__ == "__main__":
    config = PeriodicSystemConfig(
        (100,),
        (100,),
        truncation=50,
        direction=(5,),
        temperature=155,
    )
    system = SODIUM_COPPER_BRIDGE_SYSTEM_1D

    energies = np.array([2, 5, 10, 15, 20, np.inf]) * 0.001 * electron_volt

    _test_instrument_bias(system, config, energies)
