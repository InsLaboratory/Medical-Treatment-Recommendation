import os
import pickle
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from data_preprocessing import ID_COL, TIME_COL, SURVIVAL_COL, FLUID_COL, VASO_COL


# ---------------------------------------------------------------------------
# MDP hyperparameters (default values from the literature)
# ---------------------------------------------------------------------------

R_POS  = +15.0   # Positive terminal reward (survival)
R_NEG  = -15.0   # Negative terminal reward (readmission)
C_SOFA =  0.1    # Intermediate reward coefficient (SOFA shaping)
GAMMA  =  0.99   # Discount factor recommended for long ICU episodes
N_BINS =  5      # Number of discretization levels per action dimension


# ---------------------------------------------------------------------------
# 1. Transition construction
# ---------------------------------------------------------------------------

def build_mdp_dataset(
    df: pd.DataFrame,
    state_ext: list,
    r_pos: float  = R_POS,
    r_neg: float  = R_NEG,
    c_sofa: float = C_SOFA,
    scaler: Optional[MinMaxScaler] = None,
) -> dict:
    """
    Build the transition dataset (s, a, r, s', done) from the preprocessed
    and enriched DataFrame (output of build_extended_state).

    """
    # --- State matrix preparation -------------------------------------------
    imp = SimpleImputer(strategy="median")
    X_raw = imp.fit_transform(df[state_ext].values)

    if scaler is None:
        scaler = MinMaxScaler()
        S_mat = scaler.fit_transform(X_raw).astype(np.float32)
    else:
        S_mat = scaler.transform(X_raw).astype(np.float32)

    # --- Trajectory variables -----------------------------------------------
    df_sorted = df.reset_index(drop=True)
    pids      = df_sorted[ID_COL].values
    acts      = df_sorted["action_combined"].values.astype(np.int64)
    read_v    = df_sorted[SURVIVAL_COL].values
    sofa_d    = df_sorted["SOFA_delta"].values

    # --- Transition building loop -------------------------------------------
    states_list, actions_list, rewards_list = [], [], []
    next_states_list, terminals_list        = [], []

    for i in range(len(df_sorted) - 1):
        is_terminal = int(pids[i] != pids[i + 1])

        if is_terminal:
            r = r_pos if read_v[i] == 0 else r_neg
            s_next = S_mat[i]          # d3rlpy convention : s' = s at terminal
        else:
            r = -c_sofa * float(sofa_d[i])
            s_next = S_mat[i + 1]

        states_list.append(S_mat[i])
        actions_list.append(acts[i])
        rewards_list.append(r)
        next_states_list.append(s_next)
        terminals_list.append(is_terminal)

    states      = np.array(states_list,      dtype=np.float32)
    actions     = np.array(actions_list,     dtype=np.int64)
    rewards     = np.array(rewards_list,     dtype=np.float32)
    next_states = np.array(next_states_list, dtype=np.float32)
    terminals   = np.array(terminals_list,   dtype=np.int64)

    # --- Summary statistics -------------------------------------------------
    n_traj = int(terminals.sum())
    n_pos  = int(((rewards == r_pos) & (terminals == 1)).sum())
    n_neg  = int(((rewards == r_neg) & (terminals == 1)).sum())

    meta = {
        "n_transitions"     : len(states),
        "n_trajectories"    : n_traj,
        "mean_traj_length"  : round(len(states) / max(n_traj, 1), 1),
        "state_dim"         : states.shape[1],
        "n_actions"         : int(np.unique(actions).size),
        "n_terminal_pos"    : n_pos,
        "n_terminal_neg"    : n_neg,
        "imbalance_ratio"   : round(n_pos / max(n_neg, 1), 2),
        "reward_range"      : (float(rewards.min()), float(rewards.max())),
        "r_pos"             : r_pos,
        "r_neg"             : r_neg,
        "c_sofa"            : c_sofa,
        "gamma"             : GAMMA,
    }

    _print_mdp_summary(meta)

    return {
        "states"     : states,
        "actions"    : actions,
        "rewards"    : rewards,
        "next_states": next_states,
        "terminals"  : terminals,
        "scaler"     : scaler,
        "meta"       : meta,
    }


def _print_mdp_summary(meta: dict) -> None:
    """Print a readable summary of the MDP dataset."""
    print("\nMDP DATASET SUMMARY")
    print("=" * 60)
    print(f"  State dimension (S)          : {meta['state_dim']}  (STATE_EXT, normalized [0,1])")
    print(f"  Distinct actions             : {meta['n_actions']} / 25  (5×5 grid)")
    print(f"  Total transitions            : {meta['n_transitions']:,}")
    print(f"  Trajectories                 : {meta['n_trajectories']:,}  "
          f"(average length : {meta['mean_traj_length']} steps)")
    print(f"  Terminal reward +{meta['r_pos']:.0f}      : {meta['n_terminal_pos']:,}  (ReAd=0, survival)")
    print(f"  Terminal reward {meta['r_neg']:.0f}     : {meta['n_terminal_neg']:,}  (ReAd=1, readmission)")
    print(f"  Positive/negative ratio      : {meta['imbalance_ratio']:.2f}")
    print(f"  Reward range                 : [{meta['reward_range'][0]:.3f}, {meta['reward_range'][1]:.3f}]")
    print(f"  Intermediate reward          : r = −{meta['c_sofa']}×SOFA_delta")
    print(f"  Recommended γ                : {meta['gamma']}")


# ---------------------------------------------------------------------------
# 2. MDP dataset validation
# ---------------------------------------------------------------------------

def validate_mdp_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    terminals: np.ndarray,
    n_actions_max: int = 24,
) -> bool:
    
    eps = 0.01  # numerical tolerance for normalization

    assert states.shape[0] == actions.shape[0] == rewards.shape[0], \
        "Size mismatch between states, actions, rewards"
    assert states.shape == next_states.shape, \
        "states and next_states have different shapes"
    assert states.ndim == 2, "states must be 2D (N, D)"
    assert -eps <= float(states.min()) and float(states.max()) <= 1 + eps, \
        f"States outside [0,1] : min={states.min():.4f}  max={states.max():.4f}"
    assert 0 <= int(actions.min()) and int(actions.max()) <= n_actions_max, \
        f"Actions outside [0,{n_actions_max}] : min={actions.min()}  max={actions.max()}"
    assert set(np.unique(terminals)).issubset({0, 1}), \
        f"Terminals not binary : unique values = {np.unique(terminals)}"
    for arr, name in [(states, "states"), (rewards, "rewards"), (next_states, "next_states")]:
        assert not np.isnan(arr).any(), f"NaN detected in {name}"
        assert not np.isinf(arr).any(), f"Inf detected in {name}"

    print("[validate_mdp_dataset]  All consistency checks passed.")
    return True


# ---------------------------------------------------------------------------
# 3. Save artifacts
# ---------------------------------------------------------------------------

def save_mdp_artifacts(
    mdp: dict,
    state_cols_proc: list,
    state_ext: list,
    clip_bounds: dict,
    pca_model,
    n_comp_90: int,
    k_opt: int,
    output_dir: str = "preprocessed",
) -> None:
    """
    Save all artifacts required for result reproducibility 

    """
    os.makedirs(output_dir, exist_ok=True)

    # MDP dataset
    npz_path = os.path.join(output_dir, "sepsis_mdp_dataset.npz")
    np.savez_compressed(
        npz_path,
        states      = mdp["states"],
        actions     = mdp["actions"],
        rewards     = mdp["rewards"],
        next_states = mdp["next_states"],
        terminals   = mdp["terminals"],
    )

    # Metadata
    meta_path = os.path.join(output_dir, "preprocessing_metadata.pkl")
    metadata = {
        "STATE_COLS"   : state_cols_proc,
        "STATE_EXT"    : state_ext,
        "FLUID_COL"    : FLUID_COL,
        "VASO_COL"     : VASO_COL,
        "SURVIVAL_COL" : SURVIVAL_COL,
        "ID_COL"       : ID_COL,
        "TIME_COL"     : TIME_COL,
        "n_state_dim"  : mdp["states"].shape[1],
        "n_actions"    : 25,
        "N_BINS"       : N_BINS,
        "R_POS"        : mdp["meta"]["r_pos"],
        "R_NEG"        : mdp["meta"]["r_neg"],
        "C_SOFA"       : mdp["meta"]["c_sofa"],
        "gamma"        : mdp["meta"]["gamma"],
        "scaler_rl"    : mdp["scaler"],
        "clip_bounds"  : clip_bounds,
        "pca"          : pca_model,
        "N_COMP"       : n_comp_90,
        "K_OPT"        : k_opt,
        **mdp["meta"],
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    # Summary
    print("\nARTIFACTS SAVED")
    print("=" * 55)
    for fn in sorted(os.listdir(output_dir)):
        fp = os.path.join(output_dir, fn)
        print(f"  {fn:<45} {os.path.getsize(fp)/1e3:>8.1f} KB")

    print("\nCOMPLETE PIPELINE")
    print("=" * 55)
    print(f"  Base state features        : {len(state_cols_proc)}")
    print(f"  Engineered features        : {len(state_ext) - len(state_cols_proc)}")
    print(f"  Final dimension            : {mdp['states'].shape[1]}")
    print(f"  Action space size          : 25  (5×5)")
    print(f"  MDP transitions            : {len(mdp['states']):,}")
    print(f"  Trajectories               : {mdp['meta']['n_trajectories']:,} patients")
    print(f"\n  → Ready for : Behavior Cloning · CQL · BCQ · IQL (d3rlpy)")


# ---------------------------------------------------------------------------
# 4. Load saved dataset
# ---------------------------------------------------------------------------

def load_mdp_dataset(output_dir: str = "preprocessed") -> tuple[dict, dict]:
    """
    Load the MDP dataset and metadata from the output directory.

    """
    npz_path  = os.path.join(output_dir, "sepsis_mdp_dataset.npz")
    meta_path = os.path.join(output_dir, "preprocessing_metadata.pkl")

    data = np.load(npz_path)
    mdp_data = {
        "states"     : data["states"],
        "actions"    : data["actions"],
        "rewards"    : data["rewards"],
        "next_states": data["next_states"],
        "terminals"  : data["terminals"],
    }

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"[load_mdp_dataset]  {len(mdp_data['states']):,} transitions loaded  |  "
          f"state_dim={mdp_data['states'].shape[1]}")
    return mdp_data, metadata