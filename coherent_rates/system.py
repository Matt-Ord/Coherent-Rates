from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from scipy.constants import electron_volt, hbar, k
from surface_potential_analysis.basis.basis import (
    FundamentalBasis,
    FundamentalPositionBasis,
    FundamentalTransformedPositionBasis,
    FundamentalTransformedPositionBasis1d,
    TransformedPositionBasis1d,
)
from surface_potential_analysis.basis.evenly_spaced_basis import (
    EvenlySpacedBasis,
    EvenlySpacedTransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    TupleBasis,
    TupleBasisWithLengthLike,
)
from surface_potential_analysis.basis.time_basis_like import EvenlySpacedTimeBasis
from surface_potential_analysis.dynamics.schrodinger.solve import (
    solve_schrodinger_equation_diagonal,
)
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    total_surface_hamiltonian,
)
from surface_potential_analysis.operator.operations import apply_operator_to_states
from surface_potential_analysis.operator.operator import (
    apply_operator_to_state,
)
from surface_potential_analysis.potential.conversion import (
    convert_potential_to_basis,
    convert_potential_to_position_basis,
)
from surface_potential_analysis.stacked_basis.build import (
    fundamental_stacked_basis_from_shape,
)
from surface_potential_analysis.stacked_basis.conversion import (
    stacked_basis_as_fundamental_momentum_basis,
    stacked_basis_as_fundamental_position_basis,
)
from surface_potential_analysis.state_vector.plot import (
    get_periodic_x_operator,
)
from surface_potential_analysis.state_vector.state_vector import calculate_normalization
from surface_potential_analysis.state_vector.state_vector_list import (
    calculate_inner_products_elementwise,
)
from surface_potential_analysis.wavepacket.get_eigenstate import (
    get_full_bloch_hamiltonian,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    BlochWavefunctionListWithEigenvaluesList,
    generate_wavepacket,
)

if TYPE_CHECKING:
    from surface_potential_analysis.basis.explicit_basis import (
        ExplicitStackedBasisWithLength,
    )
    from surface_potential_analysis.operator.operator import (
        SingleBasisDiagonalOperator,
        SingleBasisOperator,
    )
    from surface_potential_analysis.potential.potential import Potential
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        ValueList,
    )
    from surface_potential_analysis.state_vector.state_vector import StateVector
    from surface_potential_analysis.state_vector.state_vector_list import (
        StateVectorList,
    )

_L0Inv = TypeVar("_L0Inv", bound=int)
_L1Inv = TypeVar("_L1Inv", bound=int)


@dataclass
class PeriodicSystem:
    """Represents the properties of a 1D Periodic System."""

    id: str
    """A unique ID, for use in caching"""
    barrier_energy: float
    lattice_constant: float
    mass: float


@dataclass
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int]
    resolution: tuple[int]
    n_bands: int


HYDROGEN_NICKEL_SYSTEM = PeriodicSystem(
    id="HNi",
    barrier_energy=2.5593864192e-20,
    lattice_constant=2.46e-10 / np.sqrt(2),
    mass=1.67e-27,
)

SODIUM_COPPER_SYSTEM = PeriodicSystem(
    id="NaCu",
    barrier_energy=55e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=3.8175458e-26,
)

LITHIUM_COPPER_SYSTEM = PeriodicSystem(
    id="LiCu",
    barrier_energy=45e-3 * electron_volt,
    lattice_constant=3.615e-10,
    mass=1.152414898e-26,
)


def get_potential(
    system: PeriodicSystem,
) -> Potential[TupleBasis[FundamentalTransformedPositionBasis1d[Literal[3]]]]:
    delta_x = np.sqrt(3) * system.lattice_constant / 2
    axis = FundamentalTransformedPositionBasis1d[Literal[3]](np.array([delta_x]), 3)
    vector = 0.25 * system.barrier_energy * np.array([2, -1, -1]) * np.sqrt(3)
    return {"basis": TupleBasis(axis), "data": vector}


def get_interpolated_potential(
    system: PeriodicSystem,
    resolution: tuple[_L0Inv],
) -> Potential[
    TupleBasisWithLengthLike[FundamentalTransformedPositionBasis[_L0Inv, Literal[1]]]
]:
    potential = get_potential(system)
    old = potential["basis"][0]
    basis = TupleBasis(
        TransformedPositionBasis1d[_L0Inv, Literal[3]](
            old.delta_x,
            old.n,
            resolution[0],
        ),
    )
    scaled_potential = potential["data"] * np.sqrt(resolution[0] / old.n)
    return convert_potential_to_basis(
        {"basis": basis, "data": scaled_potential},
        stacked_basis_as_fundamental_momentum_basis(basis),
    )


def get_extended_interpolated_potential(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L1Inv],
) -> Potential[
    TupleBasisWithLengthLike[
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]]
    ]
]:
    interpolated = get_interpolated_potential(system, resolution)
    old = interpolated["basis"][0]
    basis = TupleBasis(
        EvenlySpacedTransformedPositionBasis[_L1Inv, _L0Inv, Literal[0], Literal[1]](
            old.delta_x * shape[0],
            n=old.n,
            step=shape[0],
            offset=0,
        ),
    )
    scaled_potential = interpolated["data"] * np.sqrt(basis.fundamental_n / old.n)

    return {"basis": basis, "data": scaled_potential}


def _get_full_hamiltonian(
    system: PeriodicSystem,
    shape: tuple[_L0Inv],
    resolution: tuple[_L0Inv],
    *,
    bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]] | None = None,
) -> SingleBasisOperator[
    TupleBasisWithLengthLike[FundamentalPositionBasis[int, Literal[1]]],
]:
    bloch_fraction = np.array([0]) if bloch_fraction is None else bloch_fraction

    potential = get_extended_interpolated_potential(system, shape, resolution)
    converted = convert_potential_to_basis(
        potential,
        stacked_basis_as_fundamental_position_basis(potential["basis"]),
    )
    return total_surface_hamiltonian(converted, system.mass, bloch_fraction)


def _get_bloch_wavefunctions(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> BlochWavefunctionListWithEigenvaluesList[
    EvenlySpacedBasis[int, int, int],
    TupleBasisLike[FundamentalBasis[int]],
    TupleBasisWithLengthLike[FundamentalPositionBasis[int, Literal[1]]],
]:
    def hamiltonian_generator(
        bloch_fraction: np.ndarray[tuple[Literal[1]], np.dtype[np.float64]],
    ) -> SingleBasisOperator[
        TupleBasisWithLengthLike[FundamentalPositionBasis[int, Literal[1]]]
    ]:
        return _get_full_hamiltonian(
            system,
            (1,),
            config.resolution,
            bloch_fraction=bloch_fraction,
        )

    TupleBasis(FundamentalBasis)
    return generate_wavepacket(
        hamiltonian_generator,
        save_bands=EvenlySpacedBasis(config.n_bands, 1, 0),
        list_basis=fundamental_stacked_basis_from_shape(config.shape),
    )


def get_hamiltonian(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
) -> SingleBasisDiagonalOperator[ExplicitStackedBasisWithLength[Any, Any]]:
    wavefunctions = _get_bloch_wavefunctions(system, config)

    return get_full_bloch_hamiltonian(wavefunctions)


_AX0Inv = TypeVar("_AX0Inv", bound=EvenlySpacedTimeBasis[Any, Any, Any])


def solve_schrodinger_equation(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> StateVectorList[_AX0Inv, ExplicitStackedBasisWithLength[Any, Any]]:
    hamiltonian = get_hamiltonian(system, config)
    return solve_schrodinger_equation_diagonal(initial_state, times, hamiltonian)


def get_step_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    fraction: float | None,
) -> StateVector[Any]:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    basis = stacked_basis_as_fundamental_position_basis(potential["basis"])

    initial_state = {
        "basis": basis,
        "data": np.zeros(basis.n, dtype=np.complex128),
    }
    n_non_zero = 1 if fraction is None else int(fraction * basis.n)
    for i in range(n_non_zero):
        initial_state["data"][i] = 1 / np.sqrt(n_non_zero)
    return initial_state


def get_gaussian_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    fraction: float,
) -> StateVector[Any]:
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )
    basis = stacked_basis_as_fundamental_position_basis(potential["basis"])

    size = basis.n
    width = size * fraction
    data = np.zeros(basis.n, dtype=np.complex128)

    for i in range(size):
        data[i] = -np.square(i - size / 2) / (2 * width * width)

    data = np.exp(data)
    initial_state = {
        "basis": basis,
        "data": data,
    }
    initial_state["data"] = initial_state["data"] / np.sqrt(
        calculate_normalization(
            initial_state,
        ),
    )
    return initial_state


def get_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    initial_state: StateVector[Any],
    times: _AX0Inv,
    direction: tuple[int] = (1,),
) -> ValueList[_AX0Inv]:
    operator = get_periodic_x_operator(
        initial_state["basis"],
        direction=direction,
    )

    state_evolved = solve_schrodinger_equation(system, config, initial_state, times)

    state_evolved_scattered = apply_operator_to_states(operator, state_evolved)

    state_scattered = apply_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation(
        system,
        config,
        state_scattered,
        times,
    )
    return calculate_inner_products_elementwise(
        state_scattered_evolved,
        state_evolved_scattered,
    )


def get_isf_from_hamiltonian(
    hamiltonian: SingleBasisDiagonalOperator[Any],
    operator: SingleBasisOperator[Any],
    initial_state: StateVector[Any],
    times: _AX0Inv,
) -> ValueList[_AX0Inv]:
    state_evolved = solve_schrodinger_equation_diagonal(
        initial_state,
        times,
        hamiltonian,
    )

    state_evolved_scattered = apply_operator_to_states(operator, state_evolved)

    state_scattered = apply_operator_to_state(operator, initial_state)

    state_scattered_evolved = solve_schrodinger_equation_diagonal(
        state_scattered,
        times,
        hamiltonian,
    )

    return calculate_inner_products_elementwise(
        state_scattered_evolved,
        state_evolved_scattered,
    )


def get_random_boltzmann_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    temperature: float,
) -> StateVector[Any]:
    """Generate a random Boltzmann state.

    Follows the formula described in eqn 5 in
    https://doi.org/10.48550/arXiv.2002.12035.


    Args:
    ----
        system (PeriodicSystem): system
        config (PeriodicSystemConfig): config
        temperature (float): temperature of the system

    Returns:
    -------
        StateVector[Any]: state with boltzmann distribution

    """
    hamiltonian = get_hamiltonian(system, config)
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * k * temperature),
    )

    random_phase = np.exp(2j * np.pi * np.random.rand(len(hamiltonian["data"])))
    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = boltzmann_distribution * random_phase / normalization
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def get_boltzmann_state(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    temperature: float,
    phase: float,
) -> StateVector[Any]:
    hamiltonian = get_hamiltonian(system, config)
    boltzmann_distribution = np.exp(
        -hamiltonian["data"] / (2 * k * temperature),
    )

    normalization = np.sqrt(sum(np.square(boltzmann_distribution)))
    boltzmann_state = boltzmann_distribution * np.exp(1j * phase) / normalization
    return {"basis": hamiltonian["basis"][0], "data": boltzmann_state}


def get_average_boltzmann_isf(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    times: _AX0Inv,
    direction: tuple[int] = (1,),
    temperature: float = 300,
    n: int = 10,
) -> ValueList[_AX0Inv]:
    isf_data = np.zeros(times.n, dtype=np.complex128)
    hamiltonian = get_hamiltonian(system, config)
    operator = get_periodic_x_operator(hamiltonian["basis"][0], direction)

    for _i in range(n):
        isf_data += get_isf_from_hamiltonian(
            hamiltonian,
            operator,
            get_random_boltzmann_state(system, config, temperature),
            times,
        )["data"]
    isf_data = isf_data / n
    return {
        "data": isf_data,
        "basis": times,
    }


def get_cl_operator(
    system: PeriodicSystem,
    config: PeriodicSystemConfig,
    temperature: float,
) -> SingleBasisOperator[Any]:
    """Generate the operator for the stationary Caldeira-Leggett solution.

    Follows the formula from eqn 3.421 in
    https://doi.org/10.1093/acprof:oso/9780199213900.001.0001

    Args:
    ----
        system (PeriodicSystem): _description_
        config (PeriodicSystemConfig): _description_
        temperature (float): _description_

    Returns:
    -------
        SingleBasisOperator[Any]: _description_

    """
    potential = get_extended_interpolated_potential(
        system,
        config.shape,
        config.resolution,
    )

    converted_potential = convert_potential_to_position_basis(potential)
    size_pos = converted_potential["basis"].n  # size of position basis
    x_spacing = (
        system.lattice_constant * np.sqrt(3) / (2 * config.resolution[0])
    )  # size of each x interval
    m = system.mass

    data = converted_potential["data"]
    matrix = np.zeros((size_pos, size_pos), dtype=np.complex128)

    for i in range(size_pos):
        for j in range(i + 1):
            matrix[i][j] = -(data[int((i + j) / 2)] + data[int((i + j + 1) / 2)]) / (
                2 * k * temperature
            ) - (m * k * temperature * np.square(x_spacing)) * np.square(
                (i - j + (size_pos) // 2) % size_pos - size_pos // 2,
            ) / (2 * np.square(hbar))
            matrix[j][i] = matrix[i][j]
    matrix_pos = np.exp(matrix)  # density matrix in position basis (unnormalized)

    return {
        "basis": TupleBasis(converted_potential["basis"], converted_potential["basis"]),
        "data": matrix_pos,
    }
