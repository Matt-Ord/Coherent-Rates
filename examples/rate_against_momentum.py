import numpy as np
from matplotlib import pyplot as plt
from scipy.constants import Boltzmann
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis

from coherent_rates.isf import get_ak_data_1d
from coherent_rates.system import (
    SODIUM_COPPER_SYSTEM,
    SODIUM_FREE,
    PeriodicSystemConfig,
)

if __name__ == "__main__":
    config = PeriodicSystemConfig((20,), (50,), 50, temperature=155)
    system = SODIUM_COPPER_SYSTEM

    times = EvenlySpacedTimeBasis(100, 1, 0, 4e-13)

    (xdata, ydata) = get_ak_data_1d(system, config, times, n_points=3)
    plt.plot(xdata, ydata, "bo", label="NaCu")
    m, b = np.polyfit(xdata, ydata, 1)
    plt.plot(xdata, m * xdata + b, "b")
    print(xdata, ydata)
    mass = Boltzmann * config.temperature / (m * m)

    system = SODIUM_FREE
    times = EvenlySpacedTimeBasis(100, 1, 0, 2e-13)
    (xdata1, ydata1) = get_ak_data_1d(system, config, times, n_points=20)
    plt.plot(xdata1, ydata1, "ro", label="Free")
    m1, b1 = np.polyfit(xdata1, ydata1, 1)
    plt.plot(xdata1, m1 * xdata1 + b1, "r")

    mass1 = Boltzmann * config.temperature / (m1 * m1)
    print("mass=", mass)
    print("free mass=", mass1)
    print(m, b)
    print(m1, b1)
    plt.xlabel(r"$\Delta$k")
    plt.ylabel(r"$\sigma^{-1}$")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    input()
