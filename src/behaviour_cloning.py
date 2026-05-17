import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mdp_transitions(npz_path: str = "sepsis_preprocessed.npz") -> tuple:
    """
    Load states and actions from the preprocessed MDP file.

    Parameters
    ----------
    npz_path : path to the .npz file produced by mdp_builder.py

    Returns
    -------
    states  : np.ndarray, shape (n_transitions, n_features)
    actions : np.ndarray, shape (n_transitions,), integer action labels 0..24
    """
    data    = np.load(npz_path)
    states  = data["states"]
    actions = data["actions"]
    print(f"Loaded MDP transitions: states={states.shape}, actions={actions.shape}")
    return states, actions


# ---------------------------------------------------------------------------
# Data splitting and scaling
# ---------------------------------------------------------------------------

def split_and_scale_mdp(
    states: np.ndarray,
    actions: np.ndarray,
    test_size: float = 0.20,
    val_size: float  = 2 / 3,   # fraction of the non-train remainder
    random_state: int = 42,
):
    """
    Stratified train / val / test split followed by MinMax scaling
    (fitted only on the training fold to avoid data leakage).

    The default split matches the tabular baseline: 64 / 16 / 20 %.

    Returns
    -------
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
    y_train, y_val, y_test                      : np.ndarray
    scaler                                       : fitted MinMaxScaler
    """
    # Keep 70 % for train+val, 30 % for test (mirrors the tabular split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        states, actions,
        test_size=0.30,
        random_state=random_state,
        stratify=actions,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print(
        f"MDP split — Train: {X_train_scaled.shape}, "
        f"Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}"
    )
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_bc_mlp(
    hidden_layer_sizes: tuple = (256, 256),
    activation: str = "relu",
    alpha: float = 1e-4,
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPClassifier:
    """Return an untrained MLP for behaviour cloning (same arch as tabular MLP)."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate_bc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict:
    """
    Fit a behaviour-cloning MLP and return accuracy + macro F1 on the test set.

    Parameters
    ----------
    X_train / X_test : scaled feature arrays from split_and_scale_mdp
    y_train / y_test : integer action labels

    Returns
    -------
    dict with keys: model_name, accuracy, macro_f1, model
    """
    model = build_bc_mlp(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"{'Behaviour Cloning (MDP)':30s} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    return {"model_name": "Behaviour Cloning (MDP)", "accuracy": acc, "macro_f1": f1, "model": model}