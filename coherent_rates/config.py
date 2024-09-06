from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Self

import numpy as np

_DEFAULT_DIRECTION = ()


@dataclass
class PeriodicSystemConfig:
    """Configure the simlation-specific detail of the system."""

    shape: tuple[int, ...]
    resolution: tuple[int, ...]
    truncation: int | None = None
    temperature: float = field(default=150, kw_only=True)
    scattered_energy_range: tuple[float, float] = field(
        default=(-np.inf, np.inf),
        kw_only=True,
    )
    direction: tuple[int, ...] = field(default=_DEFAULT_DIRECTION, kw_only=True)

    def __post_init__(self: Self) -> None:
        if self.direction is _DEFAULT_DIRECTION:
            self.direction = tuple(0 for _ in self.shape)

    def with_direction(self: Self, direction: tuple[int, ...]) -> Self:
        copied = copy(self)
        copied.direction = direction
        return copied

    def with_temperature(self: Self, temperature: float) -> Self:
        copied = copy(self)
        copied.temperature = temperature
        return copied

    def with_resolution(self: Self, resolution: tuple[int, ...]) -> Self:
        copied = copy(self)
        copied.resolution = resolution
        return copied

    def with_shape(self: Self, shape: tuple[int, ...]) -> Self:
        copied = copy(self)
        copied.shape = shape
        return copied

    def with_truncation(self: Self, truncation: int | None) -> Self:
        copied = copy(self)
        copied.truncation = truncation
        return copied

    def with_scattered_energy_range(
        self: Self,
        energy_range: tuple[float, float],
    ) -> Self:
        copied = copy(self)
        copied.scattered_energy_range = energy_range
        return copied

    @property
    def n_bands(self: Self) -> int:
        """Total number of bands.

        Parameters
        ----------
        self : Self

        Returns
        -------
        int

        """
        return (
            np.prod(self.resolution).item()
            if self.truncation is None
            else self.truncation
        )

    def __hash__(self: Self) -> int:
        return hash(
            (
                self.shape,
                self.resolution,
                self.n_bands,
                self.temperature,
                self.direction,
                self.scattered_energy_range,
            ),
        )
