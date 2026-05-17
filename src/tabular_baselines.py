from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_logistic_regression(max_iter: int = 1000, random_state: int = 42) -> LogisticRegression:
    """Return an untrained Logistic Regression classifier."""
    return LogisticRegression(max_iter=max_iter, random_state=random_state)


def build_random_forest(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Return an untrained Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def build_mlp(
    hidden_layer_sizes: tuple = (256, 256),
    activation: str = "relu",
    alpha: float = 1e-4,
    max_iter: int = 200,
    random_state: int = 42,
) -> MLPClassifier:
    """Return an untrained MLP classifier."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Training and evaluation helpers
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Fit *model* on the training split, evaluate on the test split, and return
    a result dict with keys: model_name, accuracy, macro_f1.

    Parameters
    ----------
    model        : sklearn estimator (already instantiated)
    X_train      : scaled training features, shape (n_train, n_features)
    y_train      : integer action labels, shape (n_train,)
    X_test       : scaled test features,     shape (n_test,  n_features)
    y_test       : integer action labels,     shape (n_test,)
    model_name   : human-readable label for logging / results table
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"{model_name:30s} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

    return {"model_name": model_name, "accuracy": acc, "macro_f1": f1, "model": model}


def run_tabular_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> list[dict]:
    """
    Train all three tabular baselines and return a list of result dicts.

    Parameters
    ----------
    X_train / X_test : MinMax-scaled feature arrays
    y_train / y_test : integer action labels (0..24)
    random_state     : seed forwarded to all estimators

    Returns
    -------
    List of dicts, one per model, with keys model_name, accuracy, macro_f1, model.
    """
    baselines = [
        (build_logistic_regression(random_state=random_state), "Logistic Regression"),
        (build_random_forest(random_state=random_state),       "Random Forest"),
        (build_mlp(random_state=random_state),                 "MLP (tabular)"),
    ]

    results = []
    for model, name in baselines:
        result = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
        results.append(result)

    return results