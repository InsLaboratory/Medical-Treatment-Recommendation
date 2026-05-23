"""
iql_baseline.py
===============
Implicit Q-Learning (IQL) baseline — Sepsis Treatment.

IQL (Kostrikov et al., 2021) avoids querying out-of-distribution
actions during the Bellman backup by replacing the Q-maximisation with
expectile regression on the value function.  This makes it a
state-of-the-art offline RL algorithm without explicit constraint
penalties, and the direct "unconstrained" counterpart to CPQ-IQL.

Training objectives
-------------------
V-network  (expectile regression):
    L_V(ψ) = 𝔼_{(s,a)~D} [ L²_τ(Q(s,a) − V(s)) ]
    where L²_τ(u) = |τ − 𝟙[u<0]| · u²

Q-network (Bellman backup with frozen V):
    L_Q(θ) = 𝔼_{(s,a,r,s')~D} [(r + γ V(s') − Q(s,a))²]

Policy extraction (advantage-weighted regression):
    π(a|s) ∝ exp(β · A(s,a))   where A(s,a) = Q(s,a) − V(s)

Public API
----------
IQLTrainer
    .fit(dataloader, val_dataloader, ...)
    .get_policy() → IQLPolicy
    .save(path) / .load(path)

IQLPolicy
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
# Neural networks
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Generic two-hidden-layer MLP."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Expectile loss
# ──────────────────────────────────────────────────────────────────────────────

def expectile_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric expectile loss L²_τ(u) = |τ − 𝟙[u<0]| · u²."""
    weight = torch.where(u < 0, torch.full_like(u, 1.0 - tau), torch.full_like(u, tau))
    return (weight * u.pow(2)).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class IQLPolicy:
    """Advantage-weighted greedy policy extracted from IQL networks.

    action = argmax_a  exp(β · (Q(s,a) − V(s)))
           = argmax_a  Q(s,a) − V(s)   (monotone in a for fixed s)

    Parameters
    ----------
    q_net : MLP  (state_dim → n_actions)
    v_net : MLP  (state_dim → 1)
    beta  : float  policy temperature
    device : str
    """

    def __init__(
        self,
        q_net: MLP,
        v_net: MLP,
        beta: float = 3.0,
        device: str = "cpu",
    ) -> None:
        self._q = q_net.to(device).eval()
        self._v = v_net.to(device).eval()
        self.beta   = beta
        self.device = device

    @torch.no_grad()
    def _advantage(self, states: torch.Tensor) -> torch.Tensor:
        q = self._q(states)                      # (N, n_actions)
        v = self._v(states)                      # (N, 1)
        return q - v                             # broadcast → (N, n_actions)

    @torch.no_grad()
    def act(self, states: np.ndarray) -> np.ndarray:
        """Return argmax-advantage actions."""
        x = torch.tensor(states, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        adv = self._advantage(x)
        return adv.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return Q(s, ·) for all actions (used by Safe Actions filter)."""
        x = torch.tensor(states, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._q(x).cpu().numpy()

    def __repr__(self) -> str:
        return f"IQLPolicy(beta={self.beta})"


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class IQLTrainer:
    """Offline IQL trainer.

    Parameters
    ----------
    state_dim : int
    n_actions : int
    hidden : tuple[int, int]
    gamma : float
        Discount factor.
    tau : float
        Expectile parameter τ ∈ (0.5, 1).  Higher → more optimistic V.
    beta : float
        Policy temperature for advantage-weighted extraction.
    lr_q, lr_v : float
        Adam learning rates for Q and V networks.
    soft_update_rho : float
        Soft update rate ρ for target V-network.
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
        tau: float = 0.8,
        beta: float = 3.0,
        lr_q: float = 1e-4,
        lr_v: float = 1e-4,
        soft_update_rho: float = 0.005,
        grad_clip: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.gamma           = gamma
        self.tau_expectile   = tau
        self.beta            = beta
        self.grad_clip       = grad_clip
        self.soft_update_rho = soft_update_rho
        self.device          = device

        self.Q       = MLP(state_dim, n_actions, hidden).to(device)
        self.V       = MLP(state_dim, 1,         hidden).to(device)
        self.V_tgt   = copy.deepcopy(self.V).to(device)
        self.V_tgt.eval()

        self.opt_q = torch.optim.Adam(self.Q.parameters(), lr=lr_q)
        self.opt_v = torch.optim.Adam(self.V.parameters(), lr=lr_v)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _soft_update(self) -> None:
        rho = self.soft_update_rho
        for p, p_tgt in zip(self.V.parameters(), self.V_tgt.parameters()):
            p_tgt.data.copy_(rho * p.data + (1.0 - rho) * p_tgt.data)

    def _train_step(self, batch) -> tuple[float, float]:
        """One gradient step; returns (loss_q, loss_v)."""
        states, actions, rewards, next_states, terminals, _ = batch
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminals   = terminals.to(self.device)

        # ── V update (expectile regression) ──────────────────────────
        with torch.no_grad():
            q_all = self.Q(states)
            q_sa  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        v_s     = self.V(states).squeeze(1)
        loss_v  = expectile_loss(q_sa - v_s, self.tau_expectile)

        self.opt_v.zero_grad()
        loss_v.backward()
        nn.utils.clip_grad_norm_(self.V.parameters(), self.grad_clip)
        self.opt_v.step()

        # ── Q update (Bellman with frozen V_target) ───────────────────
        with torch.no_grad():
            v_next   = self.V_tgt(next_states).squeeze(1)
            q_target = rewards + self.gamma * (1.0 - terminals) * v_next

        q_pred  = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss_q  = F.mse_loss(q_pred, q_target)

        self.opt_q.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), self.grad_clip)
        self.opt_q.step()

        self._soft_update()
        return loss_q.item(), loss_v.item()

    @torch.no_grad()
    def _val_losses(self, val_loader) -> tuple[float, float]:
        """Validation Q-loss and V-loss."""
        total_q, total_v, n = 0.0, 0.0, 0
        self.Q.eval(); self.V.eval()
        for batch in val_loader:
            states, actions, rewards, next_states, terminals, _ = batch
            states      = states.to(self.device)
            actions     = actions.to(self.device)
            rewards     = rewards.to(self.device)
            next_states = next_states.to(self.device)
            terminals   = terminals.to(self.device)

            q_sa    = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            v_s     = self.V(states).squeeze(1)
            v_next  = self.V_tgt(next_states).squeeze(1)
            target  = rewards + self.gamma * (1.0 - terminals) * v_next

            total_q += F.mse_loss(q_sa, target).item() * len(states)
            total_v += expectile_loss(q_sa.detach() - v_s, self.tau_expectile).item() * len(states)
            n       += len(states)
        self.Q.train(); self.V.train()
        return total_q / max(n, 1), total_v / max(n, 1)

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
        """Train IQL.

        Returns
        -------
        history : list of dicts  (one per epoch)
        """
        history: List[Dict] = []
        best_val   = float("inf")
        patience_counter = 0
        best_q_state: Optional[dict] = None
        best_v_state: Optional[dict] = None

        for epoch in range(1, n_epochs + 1):
            self.Q.train(); self.V.train()
            sum_q, sum_v, n_b = 0.0, 0.0, 0
            for batch in dataloader:
                lq, lv   = self._train_step(batch)
                sum_q   += lq
                sum_v   += lv
                n_b     += 1

            train_q = sum_q / max(n_b, 1)
            train_v = sum_v / max(n_b, 1)
            val_q, val_v = self._val_losses(val_dataloader)

            rec = {
                "epoch"    : epoch,
                "train_q"  : train_q,
                "train_v"  : train_v,
                "val_q"    : val_q,
                "val_v"    : val_v,
            }
            history.append(rec)

            if val_q < best_val - min_delta:
                best_val     = val_q
                patience_counter = 0
                best_q_state = copy.deepcopy(self.Q.state_dict())
                best_v_state = copy.deepcopy(self.V.state_dict())
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(
                        {"Q": best_q_state, "V": best_v_state},
                        os.path.join(save_dir, "best_iql.pt"),
                    )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if log_every > 0:
                        print(f"  [IQL] Early stopping at epoch {epoch}  (val_q={val_q:.4f})")
                    break

            if log_every > 0 and epoch % log_every == 0:
                print(
                    f"  [IQL] epoch {epoch:>4d} | "
                    f"Q(train={train_q:.4f}, val={val_q:.4f}) | "
                    f"V(train={train_v:.4f}, val={val_v:.4f})"
                )

        if best_q_state is not None:
            self.Q.load_state_dict(best_q_state)
            self.V.load_state_dict(best_v_state)

        if save_dir:
            torch.save(
                {"Q": self.Q.state_dict(), "V": self.V.state_dict()},
                os.path.join(save_dir, "final_iql.pt"),
            )

        return history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"Q": self.Q.state_dict(), "V": self.V.state_dict()}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.Q.load_state_dict(ckpt["Q"])
        self.V.load_state_dict(ckpt["V"])
        self.V_tgt.load_state_dict(ckpt["V"])

    def get_policy(self) -> IQLPolicy:
        return IQLPolicy(
            copy.deepcopy(self.Q),
            copy.deepcopy(self.V),
            beta=self.beta,
            device=self.device,
        )
