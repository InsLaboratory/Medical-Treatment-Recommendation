"""
safe_actions.py — Stage 2 Safe Actions runtime filter
======================================================
The Safe Actions filter wraps any offline RL policy and intercepts actions
that would violate a clinical constraint at runtime.  When the proposed
action is unsafe, the filter selects the highest-advantage safe alternative.

If no safe action exists (all 25 actions violate at least one constraint),
the filter falls back to the constraint-minimising action — the one that
violates the fewest constraints, breaking ties by policy advantage.

Constraints evaluated at runtime
──────────────────────────────────
C1 — Hypotension without vasopressor (immediate, state-action check)
C2 — Metabolic deterioration without fluid (requires next state → skipped at
     decision time; evaluated retrospectively in offline metrics only)
C3 — Cumulative vasopressor over rolling 6-step window (requires history)
C4 — Abrupt vasopressor withdrawal in a critical patient (requires prev action)
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from cpq_iql import ClinicalConstraints as CC


class SafeActionsFilter:
    """
    Runtime safety wrapper for an offline RL policy.

    Parameters
    ----------
    policy    : any policy with advantages(s) -> np.ndarray of shape (1, n_actions)
    n_actions : action space size (default 25 for 5×5 fluid×vaso grid)
    window    : rolling window length for C3 cumulative dose check (default 6)
    """

    def __init__(self, policy, n_actions: int = 25, window: int = 6):
        self.policy    = policy
        self.n_actions = n_actions
        self.window    = window
        self.reset_episode()
        self._total_steps        = 0
        self._interventions      = 0
        self._all_unsafe_episodes = 0
        self._blocked_per_constraint: List[int] = [0, 0, 0, 0]

    # ──────────────────────────────────────────────────────────────────────────

    def reset_episode(self):
        """Reset per-episode state (action history, previous action)."""
        self._action_history: deque = deque([0] * self.window, maxlen=self.window)
        self._prev_action: int      = 0

    def step_done(self, action: int, next_state: Optional[np.ndarray] = None):
        """Update internal state after an action has been executed."""
        self._action_history.append(action)
        self._prev_action = action

    # ──────────────────────────────────────────────────────────────────────────

    def _is_safe(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Return binary violation vector (4,) for a single (state, action) pair.
        C2 is skipped at decision time (requires next state).
        """
        s      = state[np.newaxis]             # (1, D)
        a      = np.array([action])
        pa     = np.array([self._prev_action])
        win    = np.array([list(self._action_history) + [action]], dtype=np.int64)

        c1 = CC.c1_hypotension(s, a)[0]
        c2 = 0.0                               # C2 requires next state — deferred
        c3 = CC.c3_cumulative(win)[0]
        c4 = CC.c4_withdrawal(s, pa, a)[0]
        return np.array([c1, c2, c3, c4], dtype=np.float32)

    def safe_act(self, state: np.ndarray) -> int:
        """
        Select the highest-advantage action that violates no enforced constraint.

        If no safe action exists, select the action that minimises total violations
        (fewest violated constraints), breaking ties by policy advantage.
        """
        self._total_steps += 1

        adv      = self.policy.advantages(state[np.newaxis])[0]   # (n_actions,)
        priority = np.argsort(-adv)                                 # descending

        safe_action  = None
        best_fallback: Optional[int]  = None
        min_violations               = np.inf

        for action in priority:
            violations = self._is_safe(state, int(action))
            n_viol     = int(violations.sum())
            if n_viol == 0:
                safe_action = int(action)
                break
            if n_viol < min_violations:
                min_violations = n_viol
                best_fallback  = int(action)
                # Track which constraints caused the block
                for k, v in enumerate(violations):
                    if v > 0:
                        self._blocked_per_constraint[k] += 1

        if safe_action is None:
            # All actions violate at least one constraint
            self._interventions       += 1
            self._all_unsafe_episodes += 1
            chosen = best_fallback if best_fallback is not None else int(priority[0])
            return chosen

        # Intervention: the top-advantage action was unsafe
        if safe_action != int(priority[0]):
            self._interventions += 1

        return safe_action

    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return filter activity statistics."""
        total = max(self._total_steps, 1)
        return {
            "intervention_rate"       : self._interventions / total,
            "interventions"           : self._interventions,
            "total_steps"             : self._total_steps,
            "blocked_per_constraint"  : list(self._blocked_per_constraint),
            "all_unsafe_episodes"     : self._all_unsafe_episodes,
        }