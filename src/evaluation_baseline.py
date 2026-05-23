"""
evaluation_baseline.py
======================
Evaluation utilities for baseline policies (Random, DQN, IQL, CQL).

This module wraps the shared evaluation functions from evaluation.py and
fixes the action-array shape contract expected by ClinicalConstraints and
the Safe Actions filter.  All baselines must go through the wrappers defined
here rather than calling evaluation.py directly.

Root cause addressed
---------------------
RandomPolicy.act(state_1d) returns shape (1,) instead of a scalar.
DQN/IQL/CQLPolicy.act(state_1d) return shape (1,) from argmax.
compute_cvr_rollout builds the action array via a list comprehension
  [policy.act(s[i]) for i in range(N)]
resulting in a ragged mix of scalars and 1-element arrays that makes
np.stack / np.array raise "all input arrays must have the same shape".

Fix: PolicyWrapper normalises every act() call to return a Python int.

Public API
----------
PolicyWrapper(policy)          — adapts any baseline policy
compute_cvr_rollout_baseline   — CVR using PolicyWrapper
compute_bc_accuracy_baseline   — BC top-k using PolicyWrapper
evaluate_with_safe_actions_baseline
compute_survival_rate_baseline
compute_wis_baseline
run_fqe_baseline               — thin wrapper around FittedQEvaluator
full_evaluate_baseline         — runs the complete suite used in W4
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from evaluation import (
    _build_prev_actions,
    _build_rolling_window,
    patient_level_split,
    FittedQEvaluator,
    compute_wis_empirical_behavior,
    compute_survival_rate,
    evaluate_with_safe_actions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PolicyWrapper:
    """
    Normalise the act() / q_values() / action_probs() interface for any
    baseline policy so that every call returns the expected scalar / array.

    Parameters
    ----------
    policy    : RandomPolicy | DQNPolicy | IQLPolicy | CQLPolicy
    n_actions : int  (default 25)
    """

    def __init__(self, policy, n_actions: int = 25) -> None:
        self._policy  = policy
        self.n_actions = n_actions

    def act(self, state: np.ndarray) -> int:
        """Return a single Python int regardless of policy output shape."""
        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[np.newaxis, :]      # (1, D)
        out = self._policy.act(state)
        # out can be: ndarray shape (), (1,), (1,1), or a Python int
        return int(np.asarray(out).flat[0])

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return Q(s, ·) of shape (N, n_actions)."""
        states = np.atleast_2d(np.asarray(states, dtype=np.float32))
        q = self._policy.q_values(states)
        return np.asarray(q, dtype=np.float32).reshape(len(states), self.n_actions)

    def advantages(self, states: np.ndarray) -> np.ndarray:
        """
        Return advantage estimates A(s, ·) of shape (N, n_actions).

        For policies that expose advantages directly (IQL) we delegate.
        For Q-only policies we use A(s,a) = Q(s,a) − mean_a Q(s,a).
        """
        states = np.atleast_2d(np.asarray(states, dtype=np.float32))
        if hasattr(self._policy, 'advantages'):
            adv = self._policy.advantages(states)
        else:
            q   = self.q_values(states)
            adv = q - q.mean(axis=1, keepdims=True)
        return np.asarray(adv, dtype=np.float32)

    def action_probs(self, states: np.ndarray) -> np.ndarray:
        """
        Return π(a|s) of shape (N, n_actions) for WIS computation.

        Greedy policies assign probability 1 to argmax(Q) and 0 elsewhere.
        RandomPolicy distributes uniformly.
        """
        states = np.atleast_2d(np.asarray(states, dtype=np.float32))
        q   = self.q_values(states)
        act = q.argmax(axis=1)
        p   = np.zeros_like(q)
        p[np.arange(len(states)), act] = 1.0
        return p

    def __repr__(self) -> str:
        return f"PolicyWrapper({self._policy!r})"


# ─────────────────────────────────────────────────────────────────────────────
# CVR rollout
# ─────────────────────────────────────────────────────────────────────────────

def compute_cvr_rollout_baseline(
    policy,
    split:  dict,
    device: str = 'cpu',
) -> dict:
    """
    Compute per-constraint and total CVR for a baseline policy.

    Handles the shape normalisation that causes ValueError in the shared
    evaluation.py when policies return 1-element arrays instead of scalars.

    Parameters
    ----------
    policy : any baseline policy (or PolicyWrapper)
    split  : data split dict
    device : unused; kept for API parity with evaluation.py

    Returns
    -------
    dict — C1_hypotension, C2_metabolic, C3_cumulative, C4_withdrawal, total_cvr
    """
    from cpq_iql import ClinicalConstraints as CC

    wrapped = policy if isinstance(policy, PolicyWrapper) else PolicyWrapper(policy)

    s, sp, terms = split['states'], split['next_states'], split['terminals']

    # Build policy action array with guaranteed scalar ints
    a_pol  = np.array([wrapped.act(s[i]) for i in range(len(s))], dtype=np.int64)
    prev_a = _build_prev_actions(a_pol, terms)
    win    = _build_rolling_window(a_pol, terms, window=6)
    C      = CC.compute_all(s, sp, a_pol, prev_a, win)
    mean   = C.mean(axis=0)

    return {
        'C1_hypotension': float(mean[0]),
        'C2_metabolic'  : float(mean[1]),
        'C3_cumulative' : float(mean[2]),
        'C4_withdrawal' : float(mean[3]),
        'total_cvr'     : float(mean.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral consistency
# ─────────────────────────────────────────────────────────────────────────────

def compute_bc_accuracy_baseline(
    policy,
    split: dict,
    top_k: List[int] = None,
) -> dict:
    """
    BC top-k accuracy for a baseline policy.

    Parameters
    ----------
    policy : any baseline policy (or PolicyWrapper)
    split  : data split dict
    top_k  : k values to evaluate (default [1, 3, 5])

    Returns
    -------
    dict — top1, top3, top5, top1_norm, top3_norm, top5_norm,
           top1_argmax, fluid_top1, vaso_top1
    """
    if top_k is None:
        top_k = [1, 3, 5]

    wrapped  = policy if isinstance(policy, PolicyWrapper) else PolicyWrapper(policy)
    s        = np.asarray(split['states'], dtype=np.float32)
    a_true   = np.asarray(split['actions'], dtype=np.int64)
    n_act    = wrapped.n_actions

    all_adv  = wrapped.advantages(s)              # (N, n_actions)
    a_policy = all_adv.argmax(axis=1)

    result: dict = {}
    for k in top_k:
        top_idx = np.argsort(-all_adv, axis=1)[:, :k]
        match   = np.any(top_idx == a_true[:, None], axis=1)
        acc     = float(match.mean())
        rand    = k / n_act
        result[f'top{k}']      = acc
        result[f'top{k}_norm'] = float(np.clip((acc - rand) / max(1.0 - rand, 1e-9), 0.0, 1.0))

    result['top1_argmax'] = float((a_policy == a_true).mean())
    result['fluid_top1']  = float(((a_policy // 5) == (a_true // 5)).mean())
    result['vaso_top1']   = float(((a_policy  % 5) == (a_true  % 5)).mean())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Safe Actions filter
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_safe_actions_baseline(
    policy,
    split:     dict,
    n_actions: int = 25,
) -> dict:
    """
    Apply the Stage 2 Safe Actions filter for a baseline policy.

    Delegates to evaluation.evaluate_with_safe_actions after wrapping the
    policy so that safe_act() receives normalised action integers.

    Returns
    -------
    dict — same schema as evaluation.evaluate_with_safe_actions
    """
    wrapped = policy if isinstance(policy, PolicyWrapper) else PolicyWrapper(policy, n_actions)
    return evaluate_with_safe_actions(wrapped, split, n_actions=n_actions)


# ─────────────────────────────────────────────────────────────────────────────
# Survival Rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_survival_rate_baseline(
    policy,
    test_split:  dict,
    train_split: dict,
    device:      str = 'cpu',
    random_state: int = 42,
) -> dict:
    """
    Survival Rate for a baseline policy (delegates to evaluation.py).

    Returns
    -------
    dict — same schema as evaluation.compute_survival_rate
    """
    wrapped = policy if isinstance(policy, PolicyWrapper) else PolicyWrapper(policy)
    return compute_survival_rate(
        policy       = wrapped,
        test_split   = test_split,
        train_split  = train_split,
        device       = device,
        random_state = random_state,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_wis_baseline(
    policy,
    split:     dict,
    gamma:     float = 0.99,
    n_actions: int   = 25,
) -> dict:
    """
    Empirical WIS for a baseline policy.

    Returns
    -------
    dict — same schema as evaluation.compute_wis_empirical_behavior
    """
    wrapped = policy if isinstance(policy, PolicyWrapper) else PolicyWrapper(policy, n_actions)
    return compute_wis_empirical_behavior(
        wrapped, split, gamma=gamma, n_actions=n_actions
    )


# ─────────────────────────────────────────────────────────────────────────────
# FQE
# ─────────────────────────────────────────────────────────────────────────────

def run_fqe_baseline(
    policy,
    train_split: dict,
    val_split:   dict,
    test_split:  dict,
    state_dim:   int,
    n_actions:   int   = 25,
    gamma:       float = 0.99,
    hidden:      tuple = (256, 256),
    lr:          float = 3e-4,
    n_epochs:    int   = 200,
    patience:    int   = 20,
    batch_size:  int   = 512,
    device:      str   = 'cpu',
    seed:        int   = 42,
) -> dict:
    """
    Run Fitted Q-Evaluation for a baseline policy.

    Returns
    -------
    dict — same schema as FittedQEvaluator.evaluate()
    """
    wrapped = policy if isinstance(policy, PolicyWrapper) else PolicyWrapper(policy, n_actions)
    wis_ref = compute_wis_baseline(wrapped, test_split, gamma=gamma, n_actions=n_actions)

    fqe = FittedQEvaluator(
        state_dim  = state_dim,
        n_actions  = n_actions,
        gamma      = gamma,
        hidden     = hidden,
        lr         = lr,
        n_epochs   = n_epochs,
        patience   = patience,
        batch_size = batch_size,
        device     = device,
        seed       = seed,
    )
    fqe.fit(train_split, val_split, wrapped)
    return fqe.evaluate(test_split, wrapped, wis_ref)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation suite (used by W4_Baselines.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluate_baseline(
    policy,
    splits:      dict,
    method_name: str,
    state_dim:   int,
    n_actions:   int   = 25,
    gamma:       float = 0.99,
    device:      str   = 'cpu',
    seed:        int   = 42,
) -> dict:
    """
    Complete evaluation suite for a single baseline policy.

    Runs CVR, BC, Safe Actions, Survival Rate, WIS, and FQE on the test
    split; uses train and val splits for FQE training and SR model fitting.

    Parameters
    ----------
    policy      : any baseline policy
    splits      : dict with 'train', 'val', 'test' split dicts
    method_name : human-readable label for logging
    state_dim   : state feature dimension
    n_actions   : action space size
    gamma       : discount factor
    device      : torch device
    seed        : random seed

    Returns
    -------
    dict — nested result dict compatible with baselines_results.save_baseline_results
    """
    wrapped = PolicyWrapper(policy, n_actions)
    test    = splits['test']
    label   = method_name.upper()

    print(f'\n{"="*65}')
    print(f' Evaluating: {label}')
    print(f'{"="*65}')

    print('  [1/5] Computing CVR (policy rollout)...')
    cvr = compute_cvr_rollout_baseline(wrapped, test, device=device)

    print('  [2/5] Computing BC accuracy...')
    bca = compute_bc_accuracy_baseline(wrapped, test, top_k=[1, 3, 5])

    print('  [3/5] Applying Safe Actions filter...')
    safe = evaluate_with_safe_actions_baseline(wrapped, test, n_actions=n_actions)

    print('  [4/5] Computing Survival Rate...')
    sr = compute_survival_rate_baseline(
        wrapped, test, splits['train'], device=device, random_state=seed
    )

    print('  [5/5] Running FQE...')
    fqe = run_fqe_baseline(
        wrapped,
        train_split = splits['train'],
        val_split   = splits['val'],
        test_split  = test,
        state_dim   = state_dim,
        n_actions   = n_actions,
        gamma       = gamma,
        device      = device,
        seed        = seed,
    )

    print(f'\n  Results — {label}')
    print(f'  CVR total   : {cvr["total_cvr"]*100:.2f}%  '
          f'(C1={cvr["C1_hypotension"]*100:.2f}% '
          f'C2={cvr["C2_metabolic"]*100:.2f}% '
          f'C3={cvr["C3_cumulative"]*100:.2f}% '
          f'C4={cvr["C4_withdrawal"]*100:.2f}%)')
    print(f'  Safe CVR    : {safe["safe_total_cvr"]*100:.2f}%  '
          f'(interventions={safe["intervention_rate"]*100:.1f}%)')
    print(f'  BC  top1={bca["top1"]*100:.1f}%  top3={bca["top3"]*100:.1f}%')
    print(f'  SR  {sr["sr_policy"]*100:.2f}%  (ΔSR={sr["delta_sr"]*100:+.2f} pp)')
    print(f'  FQE V={fqe["v_fqe"]:.4f}  ΔV={fqe["delta_v"]:+.4f}')

    return {
        'cvr_rollout'  : cvr,
        'safe_actions' : safe,
        'bc_accuracy'  : bca,
        'survival_rate': sr,
        'fqe'          : fqe,
    }
