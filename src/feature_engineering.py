import numpy as np
import pandas as pd
from data_preprocessing import ID_COL, TIME_COL, FLUID_COL, VASO_COL


# ---------------------------------------------------------------------------
# 1. Derived physiological variables
# ---------------------------------------------------------------------------

def compute_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived physiological variables from raw measurements.

    """
    df = df.copy()

    df["PulsePressure"] = df["SysBP"] - df["DiaBP"]
    df["PF_ratio"]      = (df["PaO2"] / df["FiO2"].replace(0, np.nan)).clip(0, 600)
    df["BUN_Cr_ratio"]  = (df["BUN"]  / df["Creatinine"].replace(0, np.nan)).clip(0, 100)
    df["AnionGap"]      = df["Na"] - (df["Cl"] + df["HCO3"])
    df["NetBalance"]    = df["InputTotal"] - df["OutputTotal"]
    df["CumulBalance"]  = df.groupby(ID_COL)["NetBalance"].cumsum()
    df["MeanBP_delta"]  = df.groupby(ID_COL)["MeanBP"].diff().fillna(0)
    df["Lactate_delta"] = df.groupby(ID_COL)["Lactate"].diff().fillna(0)

    DERIVED_COLS = [
        "PulsePressure", "PF_ratio", "BUN_Cr_ratio", "AnionGap",
        "NetBalance", "CumulBalance", "MeanBP_delta", "Lactate_delta",
    ]
    print(f"[compute_derived_variables]  {len(DERIVED_COLS)} new features : {DERIVED_COLS}")
    return df


# ---------------------------------------------------------------------------
# 2. SOFA score
# ---------------------------------------------------------------------------

SOFA_COMPS = [
    "SOFA_resp", "SOFA_coag", "SOFA_liver",
    "SOFA_cardio", "SOFA_renal", "SOFA_neuro",
]


def compute_sofa_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the SOFA (Sequential Organ Failure Assessment) score 

    """
    df = df.copy()

    # PF_ratio required — sanity check
    if "PF_ratio" not in df.columns:
        df["PF_ratio"] = (df["PaO2"] / df["FiO2"].replace(0, np.nan)).clip(0, 600)

    # SOFA sub‑scores
    df["SOFA_resp"] = np.select(
        [df["PF_ratio"] < 100, df["PF_ratio"] < 200, df["PF_ratio"] < 300, df["PF_ratio"] < 400],
        [4, 3, 2, 1], default=0
    )
    df["SOFA_coag"] = np.select(
        [df["PlateletsCount"] < 20, df["PlateletsCount"] < 50,
         df["PlateletsCount"] < 100, df["PlateletsCount"] < 150],
        [4, 3, 2, 1], default=0
    )
    df["SOFA_liver"] = np.select(
        [df["TotalBili"] >= 12, df["TotalBili"] >= 6,
         df["TotalBili"] >= 2,  df["TotalBili"] >= 1.2],
        [4, 3, 2, 1], default=0
    )
    df["SOFA_cardio"] = np.select(
        [df[VASO_COL] > 0.1, df[VASO_COL] > 0, df["MeanBP"] < 70],
        [4, 3, 1], default=0
    )
    df["SOFA_renal"] = np.select(
        [df["Creatinine"] >= 5, df["Creatinine"] >= 3.5,
         df["Creatinine"] >= 2, df["Creatinine"] >= 1.2],
        [4, 3, 2, 1], default=0
    )
    df["SOFA_neuro"] = np.select(
        [df["GCS"] < 6, df["GCS"] < 10, df["GCS"] < 13, df["GCS"] < 15],
        [4, 3, 2, 1], default=0
    )

    # Total score and inter‑step variation
    df["SOFA_score"] = df[SOFA_COMPS].sum(axis=1)
    df["SOFA_delta"] = df.groupby(ID_COL)["SOFA_score"].diff().fillna(0)

    print(f"[compute_sofa_score]  SOFA mean={df['SOFA_score'].mean():.2f} "
          f"± {df['SOFA_score'].std():.2f}  |  SOFA≥2 : {(df['SOFA_score']>=2).mean()*100:.1f}%")
    return df


# ---------------------------------------------------------------------------
# 3. SIRS score
# ---------------------------------------------------------------------------

def compute_sirs_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 4 SIRS (Systemic Inflammatory Response Syndrome) criteria
    
    """
    df = df.copy()

    df["SIRS_temp"]     = ((df["Temp"] > 38) | (df["Temp"] < 36)).astype(int)
    df["SIRS_hr"]       = (df["HR"] > 90).astype(int)
    df["SIRS_rr"]       = (df["RR"] > 20).astype(int)
    df["SIRS_wbc"]      = ((df["WbcCount"] > 12) | (df["WbcCount"] < 4)).astype(int)
    df["SIRS_score"]    = df[["SIRS_temp", "SIRS_hr", "SIRS_rr", "SIRS_wbc"]].sum(axis=1)
    df["SIRS_positive"] = (df["SIRS_score"] >= 2).astype(int)

    print(f"[compute_sirs_score]  SIRS‑positive (≥2 criteria) : {df['SIRS_positive'].mean()*100:.1f}%")
    return df


# ---------------------------------------------------------------------------
# 4. Severity flags (Sepsis‑3)
# ---------------------------------------------------------------------------

def compute_severity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute three binary severity indicators derived from Sepsis‑3 criteria.

    """
    df = df.copy()

    df["Hyperlactatemia"]   = (df["Lactate"] >= 2).astype(int)
    df["SepticShock_proxy"] = ((df["MeanBP"] < 65) & (df[VASO_COL] > 0)).astype(int)
    df["MetabolicAcidosis"] = ((df["pH"] < 7.35) & (df["HCO3"] < 22)).astype(int)

    pct_shock = df["SepticShock_proxy"].mean() * 100
    pct_lact  = df["Hyperlactatemia"].mean() * 100
    pct_acid  = df["MetabolicAcidosis"].mean() * 100
    print(f"[compute_severity_flags]  Hyperlactatemia={pct_lact:.1f}%  "
          f"Septic shock={pct_shock:.1f}%  Acidosis={pct_acid:.1f}%")
    return df


# ---------------------------------------------------------------------------
# 5. Action discretization (5×5 grid — Health Gym convention)
# ---------------------------------------------------------------------------

N_BINS = 5


def discretize_actions(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """
    Discretize the two continuous actions into a discrete action space
    of size n_bins × n_bins = 25 

    """
    df = df.copy()

    df["fluid_bin"] = pd.qcut(
        df[FLUID_COL], q=n_bins, labels=range(n_bins), duplicates="drop"
    ).astype(int)

    df["vaso_bin"] = 0  # structural zero → bin 0
    mask_pos = df[VASO_COL] > 0
    if mask_pos.sum() > 0:
        df.loc[mask_pos, "vaso_bin"] = pd.qcut(
            df.loc[mask_pos, VASO_COL],
            q=n_bins - 1,
            labels=list(range(1, n_bins)),
            duplicates="drop",
        ).astype(int)

    df["action_combined"] = df["fluid_bin"] * n_bins + df["vaso_bin"]

    n_distinct = df["action_combined"].nunique()
    unobs = sorted(set(range(n_bins * n_bins)) - set(df["action_combined"].unique()))
    print(f"[discretize_actions]  Distinct observed actions : {n_distinct} / {n_bins**2}")
    if unobs:
        print(f"[discretize_actions]  Never observed actions (OOD) : {unobs}")
    return df


# ---------------------------------------------------------------------------
# 6. Extended state space construction
# ---------------------------------------------------------------------------

#: List of all feature names added by this module
NEW_FEATS = [
    "PulsePressure", "PF_ratio", "BUN_Cr_ratio", "AnionGap",
    "NetBalance", "CumulBalance", "MeanBP_delta", "Lactate_delta",
    "SOFA_score", "SOFA_delta",
    *SOFA_COMPS,
    "SIRS_score", "SIRS_positive",
    "Hyperlactatemia", "SepticShock_proxy", "MetabolicAcidosis",
]


def build_extended_state(
    df: pd.DataFrame,
    state_cols_proc: list,
) -> tuple[pd.DataFrame, list]:
    """
    Apply all feature engineering transformations in order and return the
    enriched DataFrame together with the extended state list STATE_EXT.

    Internal pipeline:
        compute_derived_variables → compute_sofa_score → compute_sirs_score
        → compute_severity_flags → discretize_actions

    """
    df = compute_derived_variables(df)
    df = compute_sofa_score(df)
    df = compute_sirs_score(df)
    df = compute_severity_flags(df)
    df = discretize_actions(df)

    # Consistency check
    missing_new = [f for f in NEW_FEATS if f not in df.columns]
    if missing_new:
        raise RuntimeError(
            f"[build_extended_state] Missing features : {missing_new}. "
            "Make sure compute_derived_variables is called before compute_sofa_score."
        )

    STATE_EXT = state_cols_proc + NEW_FEATS
    print(f"\n[build_extended_state]  Original state space   : {len(state_cols_proc)} features")
    print(f"[build_extended_state]  New features           : {len(NEW_FEATS)}")
    print(f"[build_extended_state]  Final dimension (STATE_EXT): {len(STATE_EXT)}")

    return df, STATE_EXT