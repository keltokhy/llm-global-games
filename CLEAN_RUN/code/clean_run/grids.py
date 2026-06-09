"""Task-cell generation for clean-run arms."""

from __future__ import annotations

from typing import Any

import numpy as np


def task_cells(arm: dict[str, Any]) -> list[dict[str, float | int]]:
    """Return deterministic country-period theta/z cells for one arm."""

    mode = arm.get("theta_grid_mode", "random_normal")
    if mode == "random_normal":
        return _random_normal_cells(arm)
    if mode == "fixed_theta_grid":
        return _fixed_theta_grid_cells(arm)
    raise ValueError(f"Unknown theta_grid_mode={mode!r}")


def _random_normal_cells(arm: dict[str, Any]) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(int(arm["seed"]))
    cells = []
    for country in range(int(arm["n_countries"])):
        z_base = rng.normal(0.0, float(arm.get("z_country_sd", 0.3)))
        for period in range(int(arm["n_periods"])):
            z_public = z_base + rng.normal(0.0, float(arm.get("z_period_sd", 0.05)))
            theta = rng.normal(z_public, float(arm.get("theta_sd", 1.0)))
            cells.append(
                {
                    "country": country,
                    "period": period,
                    "z_public": float(z_public),
                    "theta": float(theta),
                    "benefit": float(arm.get("benefit", 1.0)),
                }
            )
    return cells


def _fixed_theta_grid_cells(arm: dict[str, Any]) -> list[dict[str, float | int]]:
    theta_grid = arm.get("theta_grid")
    if not theta_grid:
        raise ValueError("fixed_theta_grid requires theta_grid")
    values = [float(v) for v in theta_grid]
    n_periods = int(arm["n_periods"])
    cells = []
    for country in range(int(arm["n_countries"])):
        for period in range(n_periods):
            theta = values[period % len(values)]
            cells.append(
                {
                    "country": country,
                    "period": period,
                    "z_public": float(arm.get("z_public", 0.0)),
                    "theta": theta,
                    "benefit": float(arm.get("benefit", 1.0)),
                }
            )
    return cells
