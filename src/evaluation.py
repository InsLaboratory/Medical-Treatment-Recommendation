"""
evaluation.py — Evaluation utilities for CPQ-IQL
=================================================
Metrics implemented
───────────────────
1. CVR (Constraint Violation Rate)        primary safety metric
2. Per-constraint CVR (C1–C4)             safety decomposition
3. Survival Rate (SR)                     primary clinical outcome
4. FQE (Fitted Q-Evaluation)              offline policy value estimator
5. BC top-k (Behavioral Consistency)      policy–clinician alignment
6. Intervention Rate                      Stage 2 filter activity
7. Empirical WIS                          secondary offline value signal

Metric compatibility across methods
─────────────────────────────────────
CVR, per-constraint CVR, SR, BC top-k, and empirical WIS are model-agnostic
and can be computed for any policy that exposes an act(s) method.
FQE can be adapted to any offline RL baseline by training the FQE Q-network
on that baseline's dataset split.

Recommended comparison table (CPQ-IQL ↔ your collaborator's method)
──────────────────────────────────────────────────────────────────────
| Metric         | Symbol  | Direction | Notes                         |
|----------------|---------|-----------|-------------------------------|
| Total CVR      | CVR     | ↓         | Primary safety metric         |
| C1–C4 CVR      | CVRₖ    | ↓         | Per-constraint breakdown      |
| Survival rate  | SR      | ↑         | Clinical outcome              |
| FQE value      | V^π_FQE | ↑         | Offline policy value estimate |
| BC top-3       | BC@3    | ↑         | Clinician alignment           |
| Empirical WIS  | WIS_emp | ↑         | Secondary; high variance      |

FQE is the standard offline RL policy evaluation metric and directly comparable
between any two methods trained on the same dataset split.  Your collaborator's
FQE estimate and yours can be compared on the shared test split.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_prev_actions(actions: np.ndarray, terminals: np.ndarray) -> np.ndarray:
    """Previous-action array, reset to 0 at each trajectory boundary."""
    prev = np.zeros_like(actions)
    prev[1:] = actions[:-1]
    boundary = np.where(terminals[:-1] == 1)[0] + 1
    prev[boundary] = 0
    return prev


def _build_rolling_window(actions: np.ndarray, terminals: np.ndarray,
                           window: int = 6) -> np.ndarray:
    """Per-trajectory rolling action window; shape (N, window)."""
    N   = len(actions)
    win = np.zeros((N, window), dtype=np.int64)
    buf, start = [0] * window, True
    for i in range(N):
        if start:
            buf = [0] * window
        buf   = buf[1:] + [int(actions[i])]
        win[i] = buf
        start  = bool(terminals[i])
    return win


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Patient-level stratified split
# ─────────────────────────────────────────────────────────────────────────────

def patient_level_split(
    states:       np.ndarray,
    actions:      np.ndarray,
    rewards:      np.ndarray,
    next_states:  np.ndarray,
    terminals:    np.ndarray,
    constraints:  np.ndarray,
    train_ratio:  float = 0.70,
    val_ratio:    float = 0.15,
    random_state: int   = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Split the dataset at patient (trajectory) level to prevent data leakage.
    Stratified by patient outcome (survival) to preserve the clinical distribution
    across splits.

    Returns a dict with keys 'train', 'val', 'test', each mapping to a dict of
    arrays with keys: states, actions, rewards, next_states, terminals, constraints.
    """
    from sklearn.model_selection import train_test_split

    traj_ends   = np.where(terminals == 1)[0]
    traj_starts = np.concatenate([[0], traj_ends[:-1] + 1])
    survival    = (rewards[traj_ends] > 0).astype(int)

    idx        = np.arange(len(traj_ends))
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    idx_tv, idx_test = train_test_split(
        idx, test_size=test_ratio, random_state=random_state, stratify=survival)
    idx_train, idx_val = train_test_split(
        idx_tv,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=random_state,
        stratify=survival[idx_tv],
    )

    def _gather(traj_indices):
        mask = np.zeros(len(states), dtype=bool)
        for ti in traj_indices:
            mask[traj_starts[ti]: traj_ends[ti] + 1] = True
        return {k: v[mask] for k, v in [
            ('states', states), ('actions', actions), ('rewards', rewards),
            ('next_states', next_states), ('terminals', terminals),
            ('constraints', constraints),
        ]}

    result = {name: _gather(idx)
              for name, idx in [('train', idx_train), ('val', idx_val), ('test', idx_test)]}

    print("Patient-level stratified split:")
    for name, split in result.items():
        n_traj = int(split['terminals'].sum())
        surv   = int((split['rewards'][split['terminals'] == 1] > 0).sum())
        print(f"  {name:5s}: {split['states'].shape[0]:>7,} transitions | "
              f"{n_traj:4d} trajectories | "
              f"survival {surv}/{n_traj} ({surv/max(n_traj,1)*100:.1f}%)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CVR via policy rollout
# ─────────────────────────────────────────────────────────────────────────────

def compute_cvr_rollout(policy, split: dict, device: str = 'cpu') -> dict:
    """
    Compute per-constraint and total Constraint Violation Rate from a policy rollout.

    The policy is rolled out over the test states; its own action history is used
    to build the prev_action array and rolling window, ensuring that C4
    (history-dependent) is evaluated correctly under the policy's own behaviour
    rather than the clinician's.

    Parameters
    ----------
    policy : PolicyWrapper with act(s) -> int and n_actions attribute
    split  : data split dict
    device : torch device string

    Returns
    -------
    dict with keys C1_hypotension, C2_metabolic, C3_cumulative, C4_withdrawal, total_cvr
    """
    from cpq_iql import ClinicalConstraints as CC

    s, sp, terms = split['states'], split['next_states'], split['terminals']
    a_pol  = np.array([policy.act(s[i]) for i in range(len(s))], dtype=np.int64)
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
# 3.  Survival rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_survival_rate(
    policy=None,
    test_split: dict = None,
    train_split: dict = None,
    device: str = 'cpu',
    random_state: int = 42,
    # Legacy positional-only call: compute_survival_rate(split)
    _legacy_split: dict = None,
) -> dict:
    """
    Survival rate — patient-level 90-day outcome metric.

    Two calling conventions are supported for backward compatibility:

    (a) Legacy (dataset-level):
            compute_survival_rate(split)
        Returns the fraction of trajectories in *split* that ended in survival
        (positive terminal reward).  Policy and train_split are ignored.

    (b) Full (policy-conditioned, notebook v2 API):
            compute_survival_rate(policy, test_split, train_split,
                                  device, random_state)
        Fits a logistic-regression outcome model on the training trajectories
        and uses it to estimate counterfactual survival probability under the
        policy's recommended action sequence on each test trajectory.

        Features used: mean of (state, action) pairs along each trajectory.
        This is a lightweight proxy — a full patient simulator would be needed
        for clinical deployment.

    Returns
    -------
    dict with keys:
        sr_clinician  : observed survival rate in test_split
        sr_policy     : model-estimated survival rate under the policy
        delta_sr      : sr_policy − sr_clinician
        ci_lo, ci_hi  : 95 % bootstrap CI on delta_sr (1000 resamples)
        n_patients    : number of test trajectories
        survival_rate : alias for sr_clinician (backward compat)
        n_trajectories: alias for n_patients   (backward compat)
        n_survived    : number of survived trajectories (clinician)
    """
    # ── Resolve legacy single-arg call ───────────────────────────────────────
    if _legacy_split is not None:
        # Called as compute_survival_rate(split) — policy is actually the split
        split = _legacy_split
        terminals = split['terminals']
        rewards   = split['rewards']
        traj_ends = np.where(terminals == 1)[0]
        if len(traj_ends) == 0:
            return {'survival_rate': float('nan'), 'sr_clinician': float('nan'),
                    'sr_policy': float('nan'), 'delta_sr': float('nan'),
                    'ci_lo': float('nan'), 'ci_hi': float('nan'),
                    'n_patients': 0, 'n_trajectories': 0, 'n_survived': 0}
        survived = int((rewards[traj_ends] > 0).sum())
        sr = survived / len(traj_ends)
        return {'survival_rate': sr, 'sr_clinician': sr, 'sr_policy': sr,
                'delta_sr': 0.0, 'ci_lo': 0.0, 'ci_hi': 0.0,
                'n_patients': len(traj_ends), 'n_trajectories': len(traj_ends),
                'n_survived': survived}

    # ── Detect legacy positional call: first arg is a dict, not a policy ─────
    if isinstance(policy, dict) and test_split is None:
        return compute_survival_rate(_legacy_split=policy)

    # ── Full policy-conditioned call ──────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    np.random.seed(random_state)

    def _traj_features(states, actions, traj_ends, traj_starts):
        """Mean (state ∥ one-hot action) per trajectory."""
        n_actions = int(actions.max()) + 1 if len(actions) > 0 else 25
        feats = []
        for ts, te in zip(traj_starts, traj_ends):
            s_t = states[ts:te+1]
            a_t = actions[ts:te+1]
            oh  = np.zeros((len(a_t), n_actions), dtype=np.float32)
            oh[np.arange(len(a_t)), a_t] = 1.0
            feats.append(np.concatenate([s_t, oh], axis=1).mean(axis=0))
        return np.array(feats, dtype=np.float32)

    def _traj_split(split):
        terms = split['terminals']
        ends  = np.where(terms == 1)[0]
        starts = np.concatenate([[0], ends[:-1] + 1])
        return ends, starts

    # ── Fit outcome model on training data (clinician actions) ───────────────
    tr_ends, tr_starts = _traj_split(train_split)
    tr_feats  = _traj_features(train_split['states'], train_split['actions'],
                                tr_ends, tr_starts)
    tr_labels = (train_split['rewards'][tr_ends] > 0).astype(int)

    clf = LogisticRegression(max_iter=500, random_state=random_state, C=0.1)
    clf.fit(tr_feats, tr_labels)

    # ── Evaluate on test data ─────────────────────────────────────────────────
    te_ends, te_starts = _traj_split(test_split)
    n_patients = len(te_ends)

    # Clinician actions (observed)
    clin_feats = _traj_features(test_split['states'], test_split['actions'],
                                 te_ends, te_starts)
    sr_clin = float(clf.predict_proba(clin_feats)[:, 1].mean())
    n_survived = int((test_split['rewards'][te_ends] > 0).sum())

    # Policy actions (counterfactual)
    pol_actions = np.array(
        [policy.act(test_split['states'][i]) for i in range(len(test_split['states']))],
        dtype=np.int64
    )
    pol_feats = _traj_features(test_split['states'], pol_actions, te_ends, te_starts)
    sr_pol = float(clf.predict_proba(pol_feats)[:, 1].mean())

    # ── Bootstrap 95 % CI on delta_sr ────────────────────────────────────────
    rng = np.random.default_rng(random_state)
    deltas = []
    for _ in range(1000):
        idx = rng.integers(0, n_patients, size=n_patients)
        deltas.append(pol_feats[idx].mean() - clin_feats[idx].mean())
    # Use model probabilities for the bootstrap
    pol_probs  = clf.predict_proba(pol_feats)[:, 1]
    clin_probs = clf.predict_proba(clin_feats)[:, 1]
    deltas_prob = []
    for _ in range(1000):
        idx = rng.integers(0, n_patients, size=n_patients)
        deltas_prob.append(float(pol_probs[idx].mean() - clin_probs[idx].mean()))
    ci_lo = float(np.percentile(deltas_prob, 2.5))
    ci_hi = float(np.percentile(deltas_prob, 97.5))

    delta = sr_pol - sr_clin
    return {
        'sr_clinician'  : sr_clin,
        'sr_policy'     : sr_pol,
        'delta_sr'      : delta,
        'ci_lo'         : ci_lo,
        'ci_hi'         : ci_hi,
        'n_patients'    : n_patients,
        # backward compat aliases
        'survival_rate' : sr_clin,
        'n_trajectories': n_patients,
        'n_survived'    : n_survived,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Fitted Q-Evaluation (FQE)
# ─────────────────────────────────────────────────────────────────────────────

class _FQENetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: tuple = (256, 256)):
        super().__init__()
        layers, prev = [], state_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def compute_fqe(
    policy,
    split:       dict,
    gamma:       float = 0.99,
    n_epochs:    int   = 200,
    batch_size:  int   = 512,
    lr:          float = 3e-4,
    hidden:      tuple = (256, 256),
    patience:    int   = 20,
    device:      str   = 'cpu',
    seed:        int   = 42,
) -> dict:
    """
    Fitted Q-Evaluation (FQE) — offline policy value estimator.

    FQE trains a Q-network Q^π to satisfy the Bellman evaluation equation
    for a fixed target policy π:

        Q^π(s, a) = r + γ · Q^π(s', π(s'))

    The estimated policy value is:

        V^π = E_{s₀ ~ d_0}[Q^π(s₀, π(s₀))]

    where d_0 is approximated by the first states of each trajectory.

    FQE is the standard offline RL policy evaluation metric.  It is directly
    comparable across methods (CPQ-IQL, CQL, BCQ, etc.) provided all methods
    are evaluated on the same held-out test split.  Your collaborator's FQE
    and yours can be directly compared if trained on the same dataset.

    Parameters
    ----------
    policy    : any policy with act(s: np.ndarray) -> int and n_actions attribute
    split     : data split dict
    gamma     : discount factor
    n_epochs  : maximum FQE training epochs
    batch_size: mini-batch size
    lr        : Adam learning rate
    hidden    : hidden layer sizes of the FQE Q-network
    patience  : early stopping patience on FQE training loss
    device    : torch device
    seed      : random seed

    Returns
    -------
    dict with keys:
        fqe_value      : mean estimated V^π across initial states
        fqe_std        : std of per-trajectory estimates
        fqe_value_list : per-trajectory value list (for bootstrap confidence intervals)
        fqe_n_traj     : number of trajectories evaluated
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    s_np  = split['states'].astype(np.float32)
    r_np  = split['rewards'].astype(np.float32)
    sp_np = split['next_states'].astype(np.float32)
    t_np  = split['terminals'].astype(np.float32)
    a_np  = split['actions'].astype(np.int64)

    state_dim = s_np.shape[1]
    n_actions = policy.n_actions

    print("  FQE: computing target policy actions on next states...")
    a_pi_sp = np.array([policy.act(sp_np[i]) for i in range(len(sp_np))], dtype=np.int64)

    S   = torch.tensor(s_np,    device=device)
    R   = torch.tensor(r_np,    device=device)
    SP  = torch.tensor(sp_np,   device=device)
    T   = torch.tensor(t_np,    device=device)
    A   = torch.tensor(a_np,    device=device)
    API = torch.tensor(a_pi_sp, device=device)

    Q_fqe = _FQENetwork(state_dim, n_actions, hidden).to(device)
    opt   = torch.optim.Adam(Q_fqe.parameters(), lr=lr)
    N     = len(S)

    best_loss, no_improve = float('inf'), 0
    print("  FQE: training evaluation Q-network...")
    for epoch in range(n_epochs):
        idx     = np.random.permutation(N)
        ep_loss = 0.0
        for start in range(0, N, batch_size):
            b            = idx[start:start + batch_size]
            s_b, r_b, sp_b, t_b = S[b], R[b], SP[b], T[b]
            a_b, api_b   = A[b], API[b]
            with torch.no_grad():
                q_next = Q_fqe(sp_b).gather(1, api_b.unsqueeze(1)).squeeze(1)
                target = r_b + gamma * (1.0 - t_b) * q_next
            q_pred = Q_fqe(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            loss   = F.mse_loss(q_pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(b)
        ep_loss /= N
        if ep_loss < best_loss - 1e-4:
            best_loss, no_improve = ep_loss, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  FQE: converged at epoch {epoch + 1}  (loss = {best_loss:.4f})")
            break

    # Estimate V^π from initial states of each trajectory
    traj_ends   = np.where(t_np == 1)[0]
    traj_starts = np.concatenate([[0], traj_ends[:-1] + 1])
    Q_fqe.eval()
    traj_values = []
    with torch.no_grad():
        for ts in traj_starts:
            s0 = S[ts].unsqueeze(0)
            a0 = policy.act(s_np[ts])
            traj_values.append(Q_fqe(s0)[0, a0].item())

    traj_values = np.array(traj_values)
    return {
        'fqe_value'     : float(np.mean(traj_values)),
        'fqe_std'       : float(np.std(traj_values)),
        'fqe_value_list': traj_values.tolist(),
        'fqe_n_traj'    : len(traj_values),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4b. FittedQEvaluator — class-based API (notebook v2)
# ─────────────────────────────────────────────────────────────────────────────

class FittedQEvaluator:
    """
    Class-based Fitted Q-Evaluation (FQE) — notebook v2 API.

    Wraps ``compute_fqe`` in a sklearn-style fit / evaluate interface so that
    the FQE network can be trained once and reused across multiple evaluate()
    calls (e.g. during sensitivity sweeps or ablation studies).

    Usage
    -----
    fqe = FittedQEvaluator(state_dim=33, n_actions=25, gamma=0.99, ...)
    fqe.fit(train_split, val_split, policy)
    result = fqe.evaluate(test_split, policy, wis_result)

    Parameters
    ----------
    state_dim  : state feature dimension
    n_actions  : number of discrete actions
    gamma      : discount factor (must match trainer)
    hidden     : hidden layer sizes of the FQE Q-network
    lr         : Adam learning rate
    n_epochs   : maximum training epochs
    patience   : early stopping patience on validation Bellman loss
    batch_size : mini-batch size
    device     : torch device string
    seed       : random seed for reproducibility
    """

    def __init__(
        self,
        state_dim:  int,
        n_actions:  int,
        gamma:      float = 0.99,
        hidden:     tuple = (256, 256),
        lr:         float = 3e-4,
        n_epochs:   int   = 200,
        patience:   int   = 20,
        batch_size: int   = 512,
        device:     str   = 'cpu',
        seed:       int   = 42,
    ):
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.hidden     = hidden
        self.lr         = lr
        self.n_epochs   = n_epochs
        self.patience   = patience
        self.batch_size = batch_size
        self.device     = device
        self.seed       = seed

        self._Q: Optional[_FQENetwork] = None
        self._n_epochs_trained: int    = 0
        self._val_bellman_loss: float  = float('nan')

    # ------------------------------------------------------------------
    def fit(self, train_split: dict, val_split: dict, policy) -> "FittedQEvaluator":
        """
        Train the FQE Q-network on *train_split* and early-stop on *val_split*.

        The validation Bellman loss is computed at the end of each epoch on a
        held-out set; this mirrors the early-stopping logic used for the main
        CPQ-IQL trainer and ensures the FQE network does not overfit.

        Parameters
        ----------
        train_split : training data split dict
        val_split   : validation data split dict (used only for early stopping)
        policy      : policy whose value is being estimated

        Returns
        -------
        self (for chaining)
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        def _prep(split):
            s_np  = split['states'].astype(np.float32)
            r_np  = split['rewards'].astype(np.float32)
            sp_np = split['next_states'].astype(np.float32)
            t_np  = split['terminals'].astype(np.float32)
            a_np  = split['actions'].astype(np.int64)
            # Target policy actions on next states
            a_pi = np.array(
                [policy.act(sp_np[i]) for i in range(len(sp_np))],
                dtype=np.int64
            )
            return (
                torch.tensor(s_np,  device=self.device),
                torch.tensor(r_np,  device=self.device),
                torch.tensor(sp_np, device=self.device),
                torch.tensor(t_np,  device=self.device),
                torch.tensor(a_np,  device=self.device),
                torch.tensor(a_pi,  device=self.device),
            )

        print("  FQE.fit: preparing training data...")
        S, R, SP, T, A, API       = _prep(train_split)
        Sv, Rv, SPv, Tv, Av, APIv = _prep(val_split)

        Q   = _FQENetwork(self.state_dim, self.n_actions, self.hidden).to(self.device)
        opt = torch.optim.Adam(Q.parameters(), lr=self.lr)
        N   = len(S)

        best_val, no_improve, best_state = float('inf'), 0, None
        print("  FQE.fit: training evaluation Q-network...")

        for epoch in range(self.n_epochs):
            Q.train()
            idx = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                b = idx[start:start + self.batch_size]
                with torch.no_grad():
                    q_next = Q(SP[b]).gather(1, API[b].unsqueeze(1)).squeeze(1)
                    target = R[b] + self.gamma * (1.0 - T[b]) * q_next
                q_pred = Q(S[b]).gather(1, A[b].unsqueeze(1)).squeeze(1)
                loss   = F.mse_loss(q_pred, target)
                opt.zero_grad(); loss.backward(); opt.step()

            # Validation Bellman loss
            Q.eval()
            with torch.no_grad():
                q_next_v = Q(SPv).gather(1, APIv.unsqueeze(1)).squeeze(1)
                target_v = Rv + self.gamma * (1.0 - Tv) * q_next_v
                q_pred_v = Q(Sv).gather(1, Av.unsqueeze(1)).squeeze(1)
                val_loss = F.mse_loss(q_pred_v, target_v).item()

            if val_loss < best_val - 1e-4:
                best_val      = val_loss
                no_improve    = 0
                best_state    = {k: v.clone() for k, v in Q.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"  FQE.fit: converged at epoch {epoch + 1}  "
                      f"(val Bellman loss = {best_val:.4f})")
                self._n_epochs_trained = epoch + 1
                break
        else:
            self._n_epochs_trained = self.n_epochs

        if best_state is not None:
            Q.load_state_dict(best_state)
        Q.eval()
        self._Q                = Q
        self._val_bellman_loss = best_val
        return self

    # ------------------------------------------------------------------
    def evaluate(self, test_split: dict, policy, wis_result: dict = None) -> dict:
        """
        Estimate V^π on *test_split* initial states using the fitted FQE network.

        Also incorporates the WIS clinician reference value (if provided) to
        compute ΔV = V_FQE − V_WIS_clinician.

        Parameters
        ----------
        test_split : test data split dict
        policy     : the same policy used in fit()
        wis_result : output of compute_wis_empirical_behavior()  (optional)

        Returns
        -------
        dict with keys:
            v_fqe            : mean FQE value over test initial states
            v_fqe_std        : std of per-trajectory values
            v_wis_clinician  : WIS clinician reference (nan if not provided)
            delta_v          : v_fqe − v_wis_clinician
            val_bellman_loss : validation Bellman loss at convergence
            n_epochs_trained : number of FQE training epochs
            fqe_value        : alias for v_fqe (backward compat with compute_fqe)
            fqe_std          : alias for v_fqe_std
        """
        if self._Q is None:
            raise RuntimeError("FittedQEvaluator.fit() must be called before evaluate().")

        s_np  = test_split['states'].astype(np.float32)
        t_np  = test_split['terminals']
        traj_ends   = np.where(t_np == 1)[0]
        traj_starts = np.concatenate([[0], traj_ends[:-1] + 1])

        S = torch.tensor(s_np, device=self.device)
        self._Q.eval()
        traj_values = []
        with torch.no_grad():
            for ts in traj_starts:
                s0 = S[ts].unsqueeze(0)
                a0 = policy.act(s_np[ts])
                traj_values.append(self._Q(s0)[0, a0].item())

        traj_values = np.array(traj_values)
        v_fqe       = float(np.mean(traj_values))
        v_fqe_std   = float(np.std(traj_values))

        v_wis = float(wis_result['wis_value']) if wis_result is not None else float('nan')
        delta = (v_fqe - v_wis) if wis_result is not None else float('nan')

        return {
            'v_fqe'           : v_fqe,
            'v_fqe_std'       : v_fqe_std,
            'v_wis_clinician' : v_wis,
            'delta_v'         : delta,
            'val_bellman_loss': self._val_bellman_loss,
            'n_epochs_trained': self._n_epochs_trained,
            # backward compat aliases (matches compute_fqe output)
            'fqe_value'       : v_fqe,
            'fqe_std'         : v_fqe_std,
            'fqe_value_list'  : traj_values.tolist(),
            'fqe_n_traj'      : len(traj_values),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Behavioral consistency (BC top-k)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bc_accuracy_rollout(
    policy, split: dict, top_k: List[int] = None
) -> dict:
    """
    Behavioral consistency via top-k advantage rank matching.

    For each state, we rank all actions by A(s,·) and check whether the
    clinician's action falls within the top-k.  A normalised score removes
    the random-baseline contribution:

        score_norm = (top_k_obs − k/n_actions) / (1 − k/n_actions)

    0 % = no better than random  ·  100 % = perfect clinician agreement.

    Parameters
    ----------
    policy : PolicyWrapper
    split  : data split dict
    top_k  : list of k values to evaluate (default [1, 3, 5])

    Returns
    -------
    dict with top1, top3, top5, top1_norm, ..., fluid_top1, vaso_top1
    """
    if top_k is None:
        top_k = [1, 3, 5]

    s, a_true = split['states'], split['actions']
    all_adv   = policy.advantages(s)           # (N, n_actions)
    a_policy  = all_adv.argmax(axis=1)
    n_act     = policy.n_actions

    result: dict = {}
    for k in top_k:
        top_idx  = np.argsort(-all_adv, axis=1)[:, :k]
        match    = np.any(top_idx == a_true[:, None], axis=1)
        acc      = float(match.mean())
        rand     = k / n_act
        result[f'top{k}']      = acc
        result[f'top{k}_norm'] = float(np.clip((acc - rand) / max(1.0 - rand, 1e-9), 0, 1))

    result['top1_argmax'] = float((a_policy == a_true).mean())
    result['fluid_top1']  = float(((a_policy // 5) == (a_true // 5)).mean())
    result['vaso_top1']   = float(((a_policy  % 5) == (a_true  % 5)).mean())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Empirical behavior policy estimation & WIS
# ─────────────────────────────────────────────────────────────────────────────

def estimate_behavior_policy(actions: np.ndarray, n_actions: int = 25) -> np.ndarray:
    """
    Estimate a marginal (state-independent) behavior policy from action frequencies.

    The state-independent estimate is a simplification; a state-conditioned
    estimate would require a density model.  The marginal estimate is substantially
    better than the uniform assumption (1/25) which over-weights action 0 by ~6×.

    Returns
    -------
    pi_b : np.ndarray of shape (n_actions,) — empirical action probabilities
    """
    counts = np.bincount(actions, minlength=n_actions).astype(np.float64)
    return counts / counts.sum()


def compute_wis_empirical_behavior(
    policy,
    split:     dict,
    gamma:     float = 0.99,
    clip:      float = 20.0,
    n_actions: int   = 25,
) -> dict:
    """
    Weighted Importance Sampling (WIS) with empirical (marginal) behavior policy.

    This replaces the uniform behavior assumption (π_b = 1/25) which caused
    importance weights to blow up to 25× for the dominant action 0.

    The marginal WIS is an approximate offline value estimate.  It is provided
    as a secondary signal alongside FQE.  FQE is preferred as the primary
    offline value estimator because it does not require importance weights.

    Parameters
    ----------
    policy    : policy with act(s) -> int and action_probs(s) -> (B, n_actions)
    split     : data split dict
    gamma     : discount factor
    clip      : per-step importance weight clip (reduces variance)
    n_actions : action space size

    Returns
    -------
    dict with keys: wis_value, wis_n_traj, wis_clip_used, behavior_policy
    """
    s_all    = split['states']
    a_all    = split['actions']
    r_all    = split['rewards']
    term_all = split['terminals']

    pi_b = estimate_behavior_policy(a_all, n_actions=n_actions)

    traj_ends   = np.where(term_all == 1)[0]
    traj_starts = np.concatenate([[0], traj_ends[:-1] + 1])

    traj_values  = []
    traj_weights = []

    for ts, te in zip(traj_starts, traj_ends):
        s_traj = s_all[ts:te+1]
        a_traj = a_all[ts:te+1]
        r_traj = r_all[ts:te+1]

        # Per-step importance ratios
        pi_pol = policy.action_probs(s_traj)    # (T, n_actions)
        ratios = np.array([
            pi_pol[t, a_traj[t]] / max(pi_b[a_traj[t]], 1e-8)
            for t in range(len(a_traj))
        ])
        ratios  = np.clip(ratios, 0.0, clip)
        cum_rho = np.cumprod(ratios)

        T       = len(r_traj)
        gammas  = np.array([gamma**t for t in range(T)])
        ret     = float((cum_rho * gammas * r_traj).sum())
        w       = float(cum_rho[-1])

        traj_values.append(ret)
        traj_weights.append(w)

    weights = np.array(traj_weights)
    values  = np.array(traj_values)
    wis_val = float((weights * values).sum() / max(weights.sum(), 1e-9))

    return {
        'wis_value'      : wis_val,
        'wis_n_traj'     : len(traj_values),
        'wis_clip_used'  : clip,
        'behavior_policy': pi_b.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Safe Actions filter evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_safe_actions(policy, split: dict, n_actions: int = 25) -> dict:
    """
    Apply the Stage 2 Safe Actions filter and compute all constraint metrics.

    The filter uses its own maintained action history for C3/C4 evaluation,
    reflecting actual filter behaviour at deployment time.

    Returns
    -------
    dict with intervention stats and per-constraint CVR after filter
    """
    from safe_actions import SafeActionsFilter
    from cpq_iql import ClinicalConstraints as CC

    sf        = SafeActionsFilter(policy, n_actions=n_actions)
    s         = split['states']
    sp        = split['next_states']
    terminals = split['terminals']
    N         = len(s)

    a_sa = np.zeros(N, dtype=np.int64)
    for i in range(N):
        if i > 0 and terminals[i - 1]:
            sf.reset_episode()
        a_sa[i] = sf.safe_act(s[i])
        sf.step_done(a_sa[i], sp[i] if i < N - 1 else None)

    prev_a = _build_prev_actions(a_sa, terminals)
    win    = _build_rolling_window(a_sa, terminals, window=6)
    C      = CC.compute_all(s, sp, a_sa, prev_a, win)
    viol   = C.mean(axis=0)
    stats  = sf.stats()

    return {
        'intervention_rate'     : stats['intervention_rate'],
        'interventions'         : stats['interventions'],
        'total_steps'           : stats['total_steps'],
        'blocked_per_constraint': stats['blocked_per_constraint'],
        'all_unsafe_episodes'   : stats['all_unsafe_episodes'],
        'safe_C1_hypotension'   : float(viol[0]),
        'safe_C2_metabolic'     : float(viol[1]),
        'safe_C3_cumulative'    : float(viol[2]),
        'safe_C4_withdrawal'    : float(viol[3]),
        'safe_total_cvr'        : float(viol.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(
    policy,
    splits:     dict,
    gamma:      float = 0.99,
    device:     str   = 'cpu',
    run_fqe:    bool  = True,
    fqe_epochs: int   = 200,
    run_wis:    bool  = True,
) -> dict:
    """
    Complete evaluation pipeline across all splits.

    Returns a nested dict: results[split_name][metric_name].
    """
    results = {}
    for name, split in splits.items():
        print(f"\n── {name} split ──")
        cvr  = compute_cvr_rollout(policy, split, device=device)
        bca  = compute_bc_accuracy_rollout(policy, split)
        safe = evaluate_with_safe_actions(policy, split)
        sr   = compute_survival_rate(split)
        row  = {**cvr, **bca, **safe, **sr}

        if run_fqe:
            print("  Running FQE...")
            fqe = compute_fqe(policy, split, gamma=gamma, device=device,
                              n_epochs=fqe_epochs)
            row.update(fqe)

        if run_wis:
            wis = compute_wis_empirical_behavior(policy, split, gamma=gamma)
            row.update(wis)

        results[name] = row
        print(
            f"  CVR={cvr['total_cvr']*100:.2f}%  "
            f"SafeCVR={safe['safe_total_cvr']*100:.2f}%  "
            f"SR={sr['survival_rate']*100:.1f}%"
            + (f"  FQE={row['fqe_value']:.3f}" if run_fqe else "")
            + f"  BC@3={bca['top3']*100:.1f}%"
        )

    return results