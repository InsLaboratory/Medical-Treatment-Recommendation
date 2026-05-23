"""
dqn_baseline.py
===============
Offline Deep Q-Network (DQN) baseline — Sepsis Treatment.

Implements a standard offline DQN trained by minimising the Bellman
residual over a fixed replay buffer.  No distribution-shift mitigation
(no CQL penalty, no behavior constraint) is applied, making this the
"vanilla" offline RL baseline.

Architecture
------------
  Q(s, a) : MLP(state_dim → hidden → hidden → n_actions)
  Q_target : hard copy updated every `target_update_freq` steps

Training objective
------------------
  L(θ) = 𝔼_{(s,a,r,s') ~ D} [(r + γ max_{a'} Q_target(s',a') − Q(s,a))²]

Public API
----------
DQNTrainer
    .fit(dataloader, val_dataloader, ...)
    .get_policy() → DQNPolicy
    .save(path) / .load(path)

DQNPolicy
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
    """Two-hidden-layer Q-network."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DQNPolicy:
    """Greedy policy derived from a trained Q-network.

    Parameters
    ----------
    q_net : QNetwork
    device : str
    """

    def __init__(self, q_net: QNetwork, device: str = "cpu") -> None:
        self._q = q_net.to(device)
        self._device = device
        self._q.eval()

    @torch.no_grad()
    def act(self, states: np.ndarray) -> np.ndarray:
        """Greedy action selection."""
        x = torch.tensor(states, dtype=torch.float32, device=self._device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        q = self._q(x)
        return q.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return Q(s, ·) for all actions."""
        x = torch.tensor(states, dtype=torch.float32, device=self._device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self._q(x).cpu().numpy()

    def __repr__(self) -> str:
        return "DQNPolicy(greedy)"


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class DQNTrainer:
    """Offline DQN trainer.

    Parameters
    ----------
    state_dim : int
    n_actions : int
    hidden : tuple[int, int]
        Hidden layer widths (default: (256, 256)).
    gamma : float
        Discount factor (default: 0.99).
    lr : float
        Adam learning rate (default: 1e-4).
    target_update_freq : int
        Steps between hard target-network updates (default: 100).
    grad_clip : float
        Max gradient norm (default: 1.0).
    device : str
    seed : int
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: tuple[int, int] = (256, 256),
        gamma: float = 0.99,
        lr: float = 1e-4,
        target_update_freq: int = 100,
        grad_clip: float = 1.0,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.gamma = gamma
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.device = device

        self.Q = QNetwork(state_dim, n_actions, hidden).to(device)
        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.Q_target.eval()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self._step = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> float:
        """Single gradient step; returns Bellman loss."""
        states, actions, rewards, next_states, terminals, _ = batch
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminals   = terminals.to(self.device)

        with torch.no_grad():
            q_next   = self.Q_target(next_states).max(dim=-1).values
            q_target = rewards + self.gamma * (1.0 - terminals) * q_next

        q_pred = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss   = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), self.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

        return loss.item()

    @torch.no_grad()
    def _val_loss(self, val_loader) -> float:
        """Compute mean Bellman loss on validation set."""
        total, n = 0.0, 0
        self.Q.eval()
        for batch in val_loader:
            states, actions, rewards, next_states, terminals, _ = batch
            states      = states.to(self.device)
            actions     = actions.to(self.device)
            rewards     = rewards.to(self.device)
            next_states = next_states.to(self.device)
            terminals   = terminals.to(self.device)

            q_next   = self.Q_target(next_states).max(dim=-1).values
            q_target = rewards + self.gamma * (1.0 - terminals) * q_next
            q_pred   = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            total   += F.mse_loss(q_pred, q_target).item() * len(states)
            n       += len(states)
        self.Q.train()
        return total / max(n, 1)

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
        """Train the DQN.

        Parameters
        ----------
        dataloader : DataLoader  (training set)
        val_dataloader : DataLoader  (validation set)
        n_epochs : int
        save_dir : str or None  — if set, best checkpoint is saved here
        log_every : int  — print every N epochs
        patience : int  — early stopping patience on val loss
        min_delta : float  — minimum improvement to reset patience counter

        Returns
        -------
        history : list of dicts  (one per epoch)
        """
        history: List[Dict] = []
        best_val  = float("inf")
        patience_counter = 0
        best_state: Optional[dict] = None

        for epoch in range(1, n_epochs + 1):
            self.Q.train()
            epoch_loss, n_batches = 0.0, 0
            for batch in dataloader:
                epoch_loss += self._train_step(batch)
                n_batches  += 1

            train_loss = epoch_loss / max(n_batches, 1)
            val_loss   = self._val_loss(val_dataloader)

            rec = {
                "epoch"      : epoch,
                "train_loss" : train_loss,
                "val_loss"   : val_loss,
            }
            history.append(rec)

            # early stopping
            if val_loss < best_val - min_delta:
                best_val     = val_loss
                patience_counter = 0
                best_state   = copy.deepcopy(self.Q.state_dict())
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(save_dir, "best_dqn.pt"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if log_every > 0:
                        print(f"  [DQN] Early stopping at epoch {epoch}  (val_loss={val_loss:.4f})")
                    break

            if log_every > 0 and epoch % log_every == 0:
                print(
                    f"  [DQN] epoch {epoch:>4d} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
                )

        # restore best weights
        if best_state is not None:
            self.Q.load_state_dict(best_state)

        if save_dir:
            torch.save(self.Q.state_dict(), os.path.join(save_dir, "final_dqn.pt"))

        return history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.Q.state_dict(), path)

    def load(self, path: str) -> None:
        self.Q.load_state_dict(torch.load(path, map_location=self.device))
        self.Q_target.load_state_dict(self.Q.state_dict())

    def get_policy(self) -> DQNPolicy:
        return DQNPolicy(copy.deepcopy(self.Q), self.device)
