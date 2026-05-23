"""
cql_baseline.py
===============
Conservative Q-Learning (CQL) baseline — Sepsis Treatment.

CQL (Kumar et al., NeurIPS 2020) addresses the distribution-shift
problem in offline RL by adding a conservative regularisation term
that penalises Q-values of actions not well-represented in the
offline dataset:

    L_CQL(θ) = α · (𝔼_{a~π} [Q(s,a)] − 𝔼_{a~πb} [Q(s,a)])
             + ½ 𝔼_{(s,a,r,s')~D} [(Q(s,a) − B^π Q(s,a))²]

In the discrete-action setting, the first term simplifies to:
    𝔼_{s~D} [ logsumexp_a Q(s,a) − Q(s, a_data) ]

This is the variant used in clinical RL literature (Tu et al., 2025).

Public API
----------
CQLTrainer
    .fit(dataloader, val_dataloader, ...)
    .get_policy() → CQLPolicy
    .save(path) / .load(path)

CQLPolicy
    .act(states) → np.ndarray
    .q_values(states) → np.ndarray
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Neural network
# ──────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Twin-head Q-network (two independent Q-functions share no weights)."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim, hidden[0]),
                nn.ReLU(),
                nn.Linear(hidden[0], hidden[1]),
                nn.ReLU(),
                nn.Linear(hidden[1], n_actions),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)

    def q_min(self, x: torch.Tensor) -> torch.Tensor:
        """Pessimistic (min) Q-value — used for policy extraction."""
        q1, q2 = self.forward(x)
        return torch.min(q1, q2)


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class CQLPolicy:
    """Greedy policy on min(Q1, Q2).

    Parameters
    ----------
    q_net : QNetwork
    device : str
    """

    def __init__(self, q_net: QNetwork, device: str = "cpu") -> None:
        self._q = q_net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def act(self, states: np.ndarray) -> np.ndarray:
        x = torch.tensor(states, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        q = self._q.q_min(x)
        return q.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return min(Q1, Q2)(s, ·)."""
        x = torch.tensor(states, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._q.q_min(x).cpu().numpy()

    def __repr__(self) -> str:
        return "CQLPolicy(greedy on min-Q)"


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class CQLTrainer:
    """Offline CQL trainer (discrete-action variant).

    Parameters
    ----------
    state_dim : int
    n_actions : int
    hidden : tuple[int, int]
    gamma : float
        Discount factor.
    alpha_cql : float
        CQL regularisation strength α.
        α = 0 recovers standard offline DQN.
        Recommended range: 0.05 – 2.0 (Tu et al., 2025 uses 0.05).
    lr : float
        Adam learning rate.
    target_update_freq : int
        Steps between hard target-network updates.
    grad_clip : float
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: tuple[int, int] = (256, 256),
        gamma: float = 0.99,
        alpha_cql: float = 0.05,
        lr: float = 1e-5,
        target_update_freq: int = 100,
        grad_clip: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.gamma            = gamma
        self.alpha_cql        = alpha_cql
        self.grad_clip        = grad_clip
        self.target_update_freq = target_update_freq
        self.device           = device
        self.n_actions        = n_actions

        self.Q      = QNetwork(state_dim, n_actions, hidden).to(device)
        self.Q_tgt  = copy.deepcopy(self.Q).to(device)
        self.Q_tgt.eval()

        # optimise both Q1 and Q2 jointly
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self._step = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cql_penalty(q_all: torch.Tensor, q_data: torch.Tensor) -> torch.Tensor:
        """Discrete CQL penalty: logsumexp_a Q(s,a) − Q(s, a_data)."""
        lse = torch.logsumexp(q_all, dim=-1)   # (N,)
        return (lse - q_data).mean()

    def _train_step(self, batch) -> dict:
        states, actions, rewards, next_states, terminals, _ = batch
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminals   = terminals.to(self.device)

        # ── Bellman target using min of target networks ───────────────
        with torch.no_grad():
            q1_next, q2_next = self.Q_tgt(next_states)
            q_next   = torch.min(q1_next, q2_next).max(dim=-1).values
            q_target = rewards + self.gamma * (1.0 - terminals) * q_next

        q1_all, q2_all = self.Q(states)
        q1_sa = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_sa = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ── Standard Bellman loss ─────────────────────────────────────
        bellman_loss = 0.5 * (F.mse_loss(q1_sa, q_target) + F.mse_loss(q2_sa, q_target))

        # ── CQL conservative penalty ──────────────────────────────────
        cql1 = self._cql_penalty(q1_all, q1_sa)
        cql2 = self._cql_penalty(q2_all, q2_sa)
        cql_loss = 0.5 * (cql1 + cql2)

        loss = bellman_loss + self.alpha_cql * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.Q_tgt.load_state_dict(self.Q.state_dict())

        return {
            "total_loss"   : loss.item(),
            "bellman_loss" : bellman_loss.item(),
            "cql_loss"     : cql_loss.item(),
        }

    @torch.no_grad()
    def _val_loss(self, val_loader) -> float:
        """Validation Bellman loss only (CQL penalty not applied at eval)."""
        total, n = 0.0, 0
        self.Q.eval()
        for batch in val_loader:
            states, actions, rewards, next_states, terminals, _ = batch
            states      = states.to(self.device)
            actions     = actions.to(self.device)
            rewards     = rewards.to(self.device)
            next_states = next_states.to(self.device)
            terminals   = terminals.to(self.device)

            q1_next, q2_next = self.Q_tgt(next_states)
            q_next   = torch.min(q1_next, q2_next).max(dim=-1).values
            q_target = rewards + self.gamma * (1.0 - terminals) * q_next

            q1_all, q2_all = self.Q(states)
            q1_sa = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_sa = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            bl = 0.5 * (F.mse_loss(q1_sa, q_target) + F.mse_loss(q2_sa, q_target))
            total += bl.item() * len(states)
            n     += len(states)

        self.Q.train()
        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def fit(
        self,
        dataloader,
        val_dataloader,
        n_epochs: int = 300,
        save_dir: Optional[str] = None,
        log_every: int = 10,
        patience: int = 40,
        min_delta: float = 1e-3,
    ) -> List[Dict]:
        """Train CQL.

        Returns
        -------
        history : list of dicts  (one per epoch)
        """
        history: List[Dict] = []
        best_val   = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None

        for epoch in range(1, n_epochs + 1):
            self.Q.train()
            sums = {"total_loss": 0.0, "bellman_loss": 0.0, "cql_loss": 0.0}
            n_b  = 0
            for batch in dataloader:
                step_out = self._train_step(batch)
                for k in sums:
                    sums[k] += step_out[k]
                n_b += 1

            means   = {k: v / max(n_b, 1) for k, v in sums.items()}
            val_loss = self._val_loss(val_dataloader)

            rec = {"epoch": epoch, "val_bellman_loss": val_loss, **means}
            history.append(rec)

            if val_loss < best_val - min_delta:
                best_val     = val_loss
                patience_counter = 0
                best_state   = copy.deepcopy(self.Q.state_dict())
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(save_dir, "best_cql.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if log_every > 0:
                        print(f"  [CQL] Early stopping at epoch {epoch}  (val_loss={val_loss:.4f})")
                    break

            if log_every > 0 and epoch % log_every == 0:
                print(
                    f"  [CQL] epoch {epoch:>4d} | "
                    f"total={means['total_loss']:.4f} | "
                    f"bellman={means['bellman_loss']:.4f} | "
                    f"cql_pen={means['cql_loss']:.4f} | "
                    f"val={val_loss:.4f}"
                )

        if best_state is not None:
            self.Q.load_state_dict(best_state)

        if save_dir:
            torch.save(self.Q.state_dict(), os.path.join(save_dir, "final_cql.pt"))

        return history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.Q.state_dict(), path)

    def load(self, path: str) -> None:
        self.Q.load_state_dict(torch.load(path, map_location=self.device))
        self.Q_tgt.load_state_dict(self.Q.state_dict())

    def get_policy(self) -> CQLPolicy:
        return CQLPolicy(copy.deepcopy(self.Q), self.device)
