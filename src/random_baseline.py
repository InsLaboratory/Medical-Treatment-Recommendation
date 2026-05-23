"""
random_baseline.py
==================
Random policy baseline for safe offline RL — Sepsis Treatment.
"""

from __future__ import annotations

import numpy as np
import torch


class RandomPolicy:
    """
    Uniform random policy over a discrete action space.

    Parameters
    ----------
    n_actions : int
        Number of discrete actions.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_actions: int = 25, seed: int = 42) -> None:
        self.n_actions = n_actions
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------

    def act(self, states: np.ndarray) -> np.ndarray:
        """
        Sample uniform random actions.

        Returns shape (N,) ALWAYS.
        """

        states = np.asarray(states)

        if states.ndim == 1:
            return np.array([self._rng.integers(0, self.n_actions)], dtype=np.int64)

        n = states.shape[0]
        return self._rng.integers(0, self.n_actions, size=n).astype(np.int64)

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """
        Uniform Q-values for compatibility with Safe Actions filter.
        Returns shape (N, n_actions).
        """

        states = np.asarray(states)

        if states.ndim == 1:
            return np.ones((1, self.n_actions), dtype=np.float32) / self.n_actions

        n = states.shape[0]
        return np.ones((n, self.n_actions), dtype=np.float32) / self.n_actions

    # ------------------------------------------------------------
    # Optional helpers
    # ------------------------------------------------------------

    def act_one(self) -> int:
        """Sample a single action."""
        return int(self._rng.integers(0, self.n_actions))

    def reset(self, seed: int | None = None):
        """Reset RNG (useful for reproducibility in rollouts)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"RandomPolicy(n_actions={self.n_actions})"