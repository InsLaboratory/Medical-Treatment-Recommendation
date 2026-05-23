"""
cpq_iql.py — CPQ-IQL for sepsis treatment
==========================================
CPQ-IQL: Constrained Pessimistic Q-learning with Implicit Q-Learning.
Two-stage safe offline RL framework:
  Stage 1 — CPQ-IQL learns a Q-function penalised by clinical constraint violations.
  Stage 2 — Safe Actions filter blocks any action that violates a hard constraint at runtime.

Action encoding (0–24):
    action    = fluid_bin * 5 + vaso_bin
    fluid_bin = action // 5   (0 = no fluid,        4 = maximum fluid)
    vaso_bin  = action  % 5   (0 = no vasopressor,  4 = maximum vasopressor)

State dimension: 56 features (Health Gym / MIMIC-III extended MDP, STATE_EXT).

Clinical constraints (Surviving Sepsis Campaign 2021):
    C1 — Hypotension without vasopressor support
    C2 — Metabolic deterioration without adequate fluid resuscitation
    C3 — Cumulative vasopressor overdose over a 6-step window
    C4 — Abrupt vasopressor withdrawal in a critically ill patient

Key design decisions:
    1. Twin Q-networks: pessimistic value via min(Q1, Q2).
    2. Penalty on advantage max(0, Q−V), not raw Q, to prevent Bellman loss collapse.
    3. Per-constraint Lagrange multiplier caps: prevents C3 from dominating.
    4. Beta annealing: policy temperature decays from beta_start to beta_end.
    5. Early stopping on validation Q-loss (not training-batch CVR, which is static).
    6. Lagrange multipliers serialised as list[float] for PyTorch >= 2.6 compatibility.
"""

import os
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Neural network components
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, out_dim: int, hidden=(256, 256), dropout=0.1) -> nn.Sequential:
    """Build a fully-connected MLP with LayerNorm and dropout after each hidden layer."""
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Discrete Q-network Q(s, a) — returns Q-values for all actions simultaneously."""

    def __init__(self, state_dim: int, n_actions: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(state_dim, n_actions, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)          # (B, n_actions)


class ValueNetwork(nn.Module):
    """State value function V(s) — scalar output per state."""

    def __init__(self, state_dim: int, hidden=(256, 256)):
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IQL loss
# ─────────────────────────────────────────────────────────────────────────────

def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Asymmetric L2 (expectile) loss from IQL (Kostrikov et al., 2021).
    tau > 0.5 produces a pessimistic value estimate conservative offline RL.
    """
    weight = torch.where(diff > 0,
                         torch.full_like(diff, tau),
                         torch.full_like(diff, 1.0 - tau))
    return (weight * diff.pow(2)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Clinical constraints
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalConstraints:
    """
    Four clinical safety constraints adapted from the Surviving Sepsis Campaign 2021
    for the 56-dimensional STATE_EXT feature space (Health Gym / MIMIC-III).

    Feature indices
    ───────────────
        Index  2: MeanBP  (normalised, μ=83 mmHg, σ=15; threshold at 65 mmHg → ~0.30)
        Index 14: Lactate (normalised)
        Index 43: SOFA_score (normalised, max raw ≈ 24)
        Index 44: SOFA_delta (normalised change)
        Index 54: SepticShock_proxy  (binary 0/1)
        Index 55: MetabolicAcidosis  (binary 0/1)

    Normalised thresholds
    ─────────────────────
        MAP_HYPOTENSION = 0.30   MAP < 65 mmHg
        MAP_BORDERLINE  = 0.40   MAP < 72 mmHg
        SOFA_CRITICAL   = 0.58   SOFA > 14 (raw), severe organ failure; only abrupt
                                 withdrawal in this range is a true safety violation.
                                 (Lower thresholds flag > 60 % of transitions, which
                                 is a dataset artefact, not a real clinical signal.)
        DELTA_SEV       = 0.10   significant SOFA deterioration
        VASO_HIGH_CUM   = 18     cumulative vaso_bin > 18 over 6 steps
    """

    MEANPB_IDX      = 2
    LACTATE_IDX     = 14
    SOFA_IDX        = 43
    SOFA_DELTA      = 44
    SEV_IDX         = 54
    ACID_IDX        = 55

    MAP_HYPOTENSION = 0.30
    MAP_BORDERLINE  = 0.40
    SOFA_CRITICAL   = 0.58      # calibrated: SOFA > 14 raw (was 0.42 → SOFA > 10,
                                 # which flagged > 60 % of all transitions)
    DELTA_SEV       = 0.10
    VASO_HIGH_CUM   = 18

    @staticmethod
    def decode_action(action: int):
        """Return (fluid_bin, vaso_bin) for combined action index 0–24."""
        return int(action) // 5, int(action) % 5

    @classmethod
    def c1_hypotension(cls, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """C1: Hypotension (MAP < 65 mmHg) without vasopressor support."""
        map_norm = states[:, cls.MEANPB_IDX]
        vaso_bin = actions % 5
        return ((map_norm < cls.MAP_HYPOTENSION) & (vaso_bin == 0)).astype(np.float32)

    @classmethod
    def c2_metabolic(cls, states: np.ndarray, next_states: np.ndarray,
                     actions: np.ndarray) -> np.ndarray:
        """C2: SOFA deterioration + borderline MAP without adequate fluid resuscitation."""
        map_norm   = states[:, cls.MEANPB_IDX]
        sofa_delta = next_states[:, cls.SOFA_IDX] - states[:, cls.SOFA_IDX]
        fluid_bin  = actions // 5
        return (
            (sofa_delta > cls.DELTA_SEV) &
            (map_norm   <= cls.MAP_BORDERLINE) &
            (fluid_bin  < 2)
        ).astype(np.float32)

    @classmethod
    def c3_cumulative(cls, actions_window: np.ndarray) -> np.ndarray:
        """
        C3: Cumulative vasopressor dose exceeds safe threshold over 6-step window.

        Parameters
        ----------
        actions_window : (N, 6) array of action integers per transition.
        """
        vaso_window = actions_window % 5
        cum_dose    = vaso_window.sum(axis=1)
        return (cum_dose > cls.VASO_HIGH_CUM).astype(np.float32)

    @classmethod
    def c4_withdrawal(cls, states: np.ndarray, prev_actions: np.ndarray,
                      actions: np.ndarray) -> np.ndarray:
        """
        C4: Abrupt vasopressor withdrawal in a severely ill patient (SOFA > 14).

        The SOFA_CRITICAL threshold is set at 0.58 (SOFA > 14 raw) to target
        only the most critical patients.  Lower thresholds create spurious violations
        because the majority of ICU transitions have SOFA in the moderate range.
        """
        sofa_norm = states[:, cls.SOFA_IDX]
        prev_vaso = prev_actions % 5
        curr_vaso = actions      % 5
        return (
            (sofa_norm > cls.SOFA_CRITICAL) &
            (prev_vaso > 0) &
            (curr_vaso == 0)
        ).astype(np.float32)

    @classmethod
    def compute_all(
        cls,
        states:         np.ndarray,
        next_states:    np.ndarray,
        actions:        np.ndarray,
        prev_actions:   np.ndarray,
        actions_window: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the full constraint matrix of shape (N, 4); entry = 1 iff violated.
        C3 requires a rolling window; if not provided it is set to zero.
        """
        c1 = cls.c1_hypotension(states, actions)
        c2 = cls.c2_metabolic(states, next_states, actions)
        c3 = cls.c3_cumulative(actions_window) if actions_window is not None \
             else np.zeros(len(states), dtype=np.float32)
        c4 = cls.c4_withdrawal(states, prev_actions, actions)
        return np.stack([c1, c2, c3, c4], axis=1)   # (N, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Offline replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class OfflineBuffer:
    """
    Wraps the preprocessed MDP .npz file and pre-computes constraint violations.

    Constraint correctness
    ──────────────────────
    C3 — built from a proper per-trajectory rolling 6-step action window.
    C4 — uses actual previous action, reset to 0 at each trajectory boundary.
    """

    def __init__(self, npz_path: str, device: str = "cpu"):
        data = np.load(npz_path)
        self.states      = data["states"].astype(np.float32)
        self.actions     = data["actions"].astype(np.int64)
        self.rewards     = data["rewards"].astype(np.float32)
        self.next_states = data["next_states"].astype(np.float32)
        self.terminals   = data["terminals"].astype(np.float32)
        self.N           = len(self.states)
        self.device      = device

        # Previous-action array — reset at trajectory boundaries
        self.prev_actions = np.zeros_like(self.actions)
        self.prev_actions[1:] = self.actions[:-1]
        term_idx = np.where(self.terminals[:-1] == 1)[0] + 1
        self.prev_actions[term_idx] = 0

        # 6-step rolling window for C3
        self.actions_window = self._build_window(window=6)

        # Pre-compute constraint violations
        self.constraints = ClinicalConstraints.compute_all(
            self.states, self.next_states,
            self.actions, self.prev_actions,
            self.actions_window,
        )   # (N, 4)

        n_viol = self.constraints.sum(axis=0)
        print(f"[OfflineBuffer]  {self.N:,} transitions loaded")
        print(f"[OfflineBuffer]  Constraint violations (clinician behavior policy):")
        names = ["C1 Hypotension", "C2 Metabolic  ", "C3 Cumulative ", "C4 Withdrawal "]
        for name, nv in zip(names, n_viol):
            print(f"                   {name}: {nv:.0f}  ({nv/self.N*100:.1f}%)")
        print(f"                   Total (mean): {n_viol.mean():.0f}  ({n_viol.mean()/self.N*100:.1f}%)")

    def _build_window(self, window: int = 6) -> np.ndarray:
        """Build per-trajectory rolling action window of shape (N, window)."""
        win  = np.zeros((self.N, window), dtype=np.int64)
        buf  = [0] * window
        traj_start = True
        for i in range(self.N):
            if traj_start:
                buf = [0] * window
            buf = buf[1:] + [int(self.actions[i])]
            win[i] = buf
            traj_start = bool(self.terminals[i])
        return win

    def make_dataloader(self, batch_size: int = 512, shuffle: bool = True) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(self.states),
            torch.from_numpy(self.actions),
            torch.from_numpy(self.rewards),
            torch.from_numpy(self.next_states),
            torch.from_numpy(self.terminals),
            torch.from_numpy(self.constraints),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CPQ-IQL Trainer
# ─────────────────────────────────────────────────────────────────────────────

class CPQIQLTrainer:
    """
    CPQ-IQL: Constrained Pessimistic Q-learning + Implicit Q-Learning.

    Architecture
    ────────────
    Twin Q-networks Q1, Q2  (pessimistic via min)
    Value network V          (expectile regression)
    Target value network V_target  (soft-updated)

    Constraint handling
    ───────────────────
    Lagrangian dual: for each constraint k, a multiplier λ_k is maintained.
    The penalty is applied to the advantage A(s,a) = max(0, Q(s,a) − V(s))
    rather than raw Q(s,a) to prevent the penalty from growing with Q magnitude
    and destroying the Bellman loss signal.

    Each λ_k is capped at lambda_max[k].  The default caps [10, 10, 8, 10]
    prevent the C3 (cumulative vasopressor) term from dominating.

    Training stability
    ──────────────────
    Beta annealing: starts at beta (high, sharp policy) and decays to beta_end.
    Early stopping: monitors validation Q-loss; training-batch CVR is a static
    dataset property and must not be used as a stopping criterion.
    """

    def __init__(
        self,
        state_dim:             int,
        n_actions:             int   = 25,
        hidden:                tuple = (256, 256),
        gamma:                 float = 0.99,
        tau:                   float = 0.8,
        beta:                  float = 5.0,
        beta_anneal:           bool  = True,
        beta_end:              float = 1.0,
        alpha:                 float = 1.0,
        eta:                   float = 0.01,
        lambda_init:           float = 1.0,
        lambda_max: Union[float, List[float]] = None,
        constraint_tol:        list  = None,
        penalty_on_advantage:  bool  = True,
        lr_q:                  float = 1e-4,
        lr_v:                  float = 1e-4,
        soft_update_rho:       float = 0.005,
        grad_clip:             float = 1.0,
        device:                str   = None,
        seed:                  int   = 42,
    ):
        self.gamma    = gamma
        self.tau      = tau
        self.beta     = beta
        self.beta_anneal  = beta_anneal
        self.beta_end     = beta_end
        self._beta_start  = beta
        self.alpha    = alpha
        self.eta      = eta
        self.grad_clip    = grad_clip
        self.rho          = soft_update_rho
        self.n_actions    = n_actions
        self.penalty_on_advantage = penalty_on_advantage
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Per-constraint lambda caps
        if lambda_max is None:
            self._lambda_max = np.array([10.0, 10.0, 8.0, 10.0])
        elif isinstance(lambda_max, (int, float)):
            self._lambda_max = np.full(4, float(lambda_max))
        else:
            self._lambda_max = np.array(lambda_max, dtype=np.float64)

        # Constraint tolerances (0 = hard constraint, 0.05 = 5% slack for C2)
        self.tol = np.array(constraint_tol or [0.0, 0.05, 0.0, 0.0])

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.Q        = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.Q2       = QNetwork(state_dim, n_actions, hidden).to(self.device)
        self.V        = ValueNetwork(state_dim, hidden).to(self.device)
        self.V_target = ValueNetwork(state_dim, hidden).to(self.device)
        self.V_target.load_state_dict(self.V.state_dict())

        self.opt_Q = torch.optim.Adam(
            list(self.Q.parameters()) + list(self.Q2.parameters()), lr=lr_q)
        self.opt_V = torch.optim.Adam(self.V.parameters(), lr=lr_v)

        # Lagrange multipliers — stored as list[float] for PyTorch >= 2.6 compatibility
        self.lambdas = np.full(4, lambda_init, dtype=np.float64)
        self.history: list = []

    # ─────────────────────────────────────────────────────────────────────────

    def _soft_update(self):
        for tp, p in zip(self.V_target.parameters(), self.V.parameters()):
            tp.data.mul_(1 - self.rho).add_(p.data * self.rho)

    def _anneal_beta(self, epoch: int, total_epochs: int):
        if not self.beta_anneal:
            return
        frac = min(epoch / max(total_epochs, 1), 1.0)
        self.beta = self._beta_start + frac * (self.beta_end - self._beta_start)

    # ─────────────────────────────────────────────────────────────────────────

    def train_epoch(self, dataloader) -> dict:
        """One full pass through the offline dataset."""
        self.Q.train(); self.Q2.train(); self.V.train()
        total = dict(loss_q=0., loss_v=0., loss_total=0., n=0,
                     viol_c1=0., viol_c2=0., viol_c3=0., viol_c4=0.)

        for batch in dataloader:
            s, a, r, sp, done, constr = [x.to(self.device) for x in batch]
            s = s.float(); sp = sp.float(); r = r.float()
            done = done.float(); a = a.long(); constr = constr.float()
            B = s.size(0)

            # Value update — expectile regression on min(Q1, Q2)
            with torch.no_grad():
                q1_sa = self.Q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                q2_sa = self.Q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
                q_min = torch.min(q1_sa, q2_sa)
            v_s    = self.V(s)
            loss_v = expectile_loss(q_min - v_s, self.tau)
            self.opt_V.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.V.parameters(), self.grad_clip)
            self.opt_V.step()

            # Q update — Bellman + Lagrangian penalty
            with torch.no_grad():
                v_sp      = self.V_target(sp)
                td_target = r + self.gamma * (1.0 - done) * v_sp

            q1_all = self.Q(s); q2_all = self.Q2(s)
            q1_sa  = q1_all.gather(1, a.unsqueeze(1)).squeeze(1)
            q2_sa  = q2_all.gather(1, a.unsqueeze(1)).squeeze(1)
            loss_q = F.mse_loss(q1_sa, td_target) + F.mse_loss(q2_sa, td_target)

            penalty = 0.0
            if self.alpha > 0:
                with torch.no_grad():
                    v_s_det = self.V(s)
                q_pen = torch.min(q1_sa, q2_sa)
                adv_pen = torch.clamp(q_pen - v_s_det, min=0.0) \
                          if self.penalty_on_advantage else q_pen
                for k in range(4):
                    penalty += float(self.lambdas[k]) * (constr[:, k] * adv_pen).mean()

            loss_total = loss_q + self.alpha * penalty
            self.opt_Q.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(
                list(self.Q.parameters()) + list(self.Q2.parameters()), self.grad_clip)
            self.opt_Q.step()

            # Dual variable update
            c_bar = constr.mean(dim=0).cpu().numpy()
            for k in range(4):
                self.lambdas[k] = np.clip(
                    self.lambdas[k] + self.eta * (c_bar[k] - self.tol[k]),
                    0.0, self._lambda_max[k])

            self._soft_update()

            total['loss_q']     += loss_q.item() * B
            total['loss_v']     += loss_v.item() * B
            total['loss_total'] += loss_total.item() * B
            total['viol_c1']    += float(c_bar[0]) * B
            total['viol_c2']    += float(c_bar[1]) * B
            total['viol_c3']    += float(c_bar[2]) * B
            total['viol_c4']    += float(c_bar[3]) * B
            total['n']          += B

        n = total.pop('n')
        return {k: v / n for k, v in total.items()}

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def eval_q_loss(self, dataloader) -> float:
        """
        Mean Q-loss on a held-out validation split.
        Used for early stopping — validation Q-loss decreases as the Q-function
        converges, making it a reliable stopping criterion (unlike training-batch CVR,
        which is a fixed dataset property that cannot change during training).
        """
        self.Q.eval(); self.Q2.eval(); self.V_target.eval()
        total_loss, total_n = 0.0, 0
        for batch in dataloader:
            s, a, r, sp, done, _ = [x.to(self.device) for x in batch]
            s = s.float(); sp = sp.float(); r = r.float()
            done = done.float(); a = a.long()
            B = s.size(0)
            v_sp      = self.V_target(sp)
            td_target = r + self.gamma * (1.0 - done) * v_sp
            q1_sa = self.Q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            q2_sa = self.Q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss  = F.mse_loss(q1_sa, td_target) + F.mse_loss(q2_sa, td_target)
            total_loss += loss.item() * B
            total_n    += B
        return total_loss / max(total_n, 1)

    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        dataloader,
        val_dataloader   = None,
        n_epochs:  int   = 300,
        save_dir:  str   = "models/cpq_iql",
        log_every: int   = 10,
        patience:  int   = 40,
        min_delta: float = 1e-3,
    ) -> list:
        """
        Full training loop with validation-based early stopping.

        Parameters
        ----------
        dataloader     : training DataLoader
        val_dataloader : validation DataLoader for early stopping; if None,
                         early stopping is disabled.
        n_epochs       : maximum number of training epochs
        save_dir       : directory to save checkpoints
        log_every      : print interval (epochs)
        patience       : epochs without val Q-loss improvement before stopping
        min_delta      : minimum improvement to count as progress
        """
        os.makedirs(save_dir, exist_ok=True)
        if val_dataloader is None:
            print("[WARNING] No validation dataloader provided — early stopping disabled.")

        best_val_loss = float("inf")
        best_epoch    = 0
        no_improve    = 0

        for epoch in range(1, n_epochs + 1):
            self._anneal_beta(epoch - 1, n_epochs)
            t0   = time.time()
            logs = self.train_epoch(dataloader)

            val_q_loss = self.eval_q_loss(val_dataloader) \
                         if val_dataloader is not None else logs['loss_q']

            logs['epoch']      = epoch
            logs['beta']       = self.beta
            logs['val_q_loss'] = val_q_loss
            logs['cvr_batch']  = (logs['viol_c1'] + logs['viol_c2'] +
                                   logs['viol_c3'] + logs['viol_c4']) / 4.0
            for k, lv in enumerate(self.lambdas):
                logs[f'lambda_{k+1}'] = float(lv)
            self.history.append(logs)

            if epoch % log_every == 0:
                print(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"Q:{logs['loss_q']:.4f}  V:{logs['loss_v']:.4f}  "
                    f"Val-Q:{val_q_loss:.4f} | "
                    f"C1:{logs['viol_c1']*100:.1f}%  "
                    f"C2:{logs['viol_c2']*100:.1f}%  "
                    f"C3:{logs['viol_c3']*100:.1f}%  "
                    f"C4:{logs['viol_c4']*100:.1f}% | "
                    f"β={self.beta:.2f}  "
                    f"λ=[{','.join(f'{l:.2f}' for l in self.lambdas)}] | "
                    f"{time.time()-t0:.1f}s"
                )

            if val_q_loss < best_val_loss - min_delta:
                best_val_loss = val_q_loss
                best_epoch    = epoch
                no_improve    = 0
                self.save(os.path.join(save_dir, "best_cpq_iql.pt"))
            else:
                no_improve += 1

            if val_dataloader is not None and no_improve >= patience:
                print(f"\n[Early stopping] No improvement for {patience} epochs. "
                      f"Best epoch: {best_epoch}  (val Q-loss = {best_val_loss:.4f})")
                break

        self.save(os.path.join(save_dir, "final_cpq_iql.pt"))
        print(f"\nTraining complete.  Best epoch: {best_epoch}  "
              f"(val Q-loss = {best_val_loss:.4f})")
        return self.history

    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save model checkpoint.
        Lambdas are serialised as list[float] — compatible with PyTorch >= 2.6
        weights_only=True mode.
        """
        def _sanitize(obj):
            if isinstance(obj, dict):   return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):   return [_sanitize(v) for v in obj]
            if hasattr(obj, 'item'):    return obj.item()
            return obj

        torch.save({
            'Q'       : self.Q.state_dict(),
            'Q2'      : self.Q2.state_dict(),
            'V'       : self.V.state_dict(),
            'V_target': self.V_target.state_dict(),
            'lambdas' : [float(l) for l in self.lambdas],
            'beta'    : float(self.beta),
            'history' : _sanitize(self.history),
        }, path)

    def load(self, path: str):
        """
        Load checkpoint.  Supports both old (numpy scalar) and new (list) lambda formats.
        """
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.Q.load_state_dict(ckpt['Q'])
        self.Q2.load_state_dict(ckpt['Q2'])
        self.V.load_state_dict(ckpt['V'])
        self.V_target.load_state_dict(ckpt['V_target'])
        raw = ckpt['lambdas']
        self.lambdas = np.array([float(x) for x in raw], dtype=np.float64) \
                       if hasattr(raw, '__len__') else \
                       np.full(4, float(raw), dtype=np.float64)
        if 'beta' in ckpt:
            self.beta = float(ckpt['beta'])
        self.history = ckpt.get('history', [])
        print(f"[load]  Checkpoint loaded: {path}")
        print(f"[load]  λ = {[f'{l:.4f}' for l in self.lambdas]}   β = {self.beta:.4f}")
        return self

    # ─────────────────────────────────────────────────────────────────────────

    def get_policy(self):
        """Return a PolicyWrapper for inference."""
        return PolicyWrapper(self.Q, self.Q2, self.V, self.n_actions, self.beta,
                             device=self.device)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Policy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PolicyWrapper:
    """
    Inference wrapper for the learned CPQ-IQL policy.

    Action selection
    ────────────────
    π(a|s) ∝ exp(β · A(s,a))   where A(s,a) = min(Q1,Q2)(s,a) − V(s)

    Behavioral consistency
    ──────────────────────
    BC accuracy is measured via top-k rank matching: the policy advantage A(s,·)
    is ranked over all actions, and we check whether the clinician's action falls
    in the top-k.  Top-1 argmax accuracy is misleading because action 0
    (no-fluid / no-vaso) accounts for ~24 % of clinician transitions.
    """

    def __init__(self, Q, Q2, V, n_actions, beta, device="cpu"):
        self.Q         = Q.eval()
        self.Q2        = Q2.eval()
        self.V         = V.eval()
        self.n_actions = n_actions
        self.beta      = beta
        self.device    = device

    @torch.no_grad()
    def advantages(self, s: np.ndarray) -> np.ndarray:
        """A(s,a) = min(Q1,Q2)(s,a) − V(s) for all actions; shape (B, n_actions)."""
        st = torch.tensor(s, dtype=torch.float32, device=self.device)
        if st.ndim == 1:
            st = st.unsqueeze(0)
        q_min = torch.min(self.Q(st), self.Q2(st))
        v     = self.V(st).unsqueeze(1)
        return (q_min - v).cpu().numpy()

    @torch.no_grad()
    def act(self, s: np.ndarray, deterministic: bool = True) -> int:
        adv = self.advantages(s)[0]
        if deterministic:
            return int(np.argmax(adv))
        adv_shifted = adv - adv.max()
        probs = np.exp(self.beta * adv_shifted)
        probs /= probs.sum()
        return int(np.random.choice(self.n_actions, p=probs))

    @torch.no_grad()
    def action_probs(self, s: np.ndarray) -> np.ndarray:
        """Softmax policy probabilities; shape (B, n_actions)."""
        adv = self.advantages(s)
        adv = adv - adv.max(axis=1, keepdims=True)
        exp_adv = np.exp(self.beta * adv)
        return exp_adv / exp_adv.sum(axis=1, keepdims=True)

    @torch.no_grad()
    def top_k_actions(self, s: np.ndarray, k: int = 3) -> np.ndarray:
        """Top-k action indices by advantage for each state; shape (B, k)."""
        adv = self.advantages(s)
        return np.argsort(-adv, axis=1)[:, :k]