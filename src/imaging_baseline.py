import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image


# ---------------------------------------------------------------------------
# Constants (must stay in sync with data_preprocessing.py)
# ---------------------------------------------------------------------------

STATE_COLS = [
    "HR", "SysBP", "MeanBP", "DiaBP", "RR", "Temp", "SpO2", "GCS",
    "pH", "BE", "HCO3", "FiO2", "PaO2", "PaCO2", "Lactate",
    "K", "Na", "Cl", "Ca", "IonisedCa", "CO2", "Mg",
    "BUN", "Creatinine", "Albumin", "SGOT", "SGPT", "TotalBili",
    "Hb", "WbcCount", "PlateletsCount", "PTT", "PT", "INR", "Age",
]

# Column names — keep consistent with data_preprocessing.py
ID_COL     = "PatientID"
TIME_COL   = "Timepoints"
ACTION_COL = "action_combined"   # created by discretize_actions() in feature_engineering.py

T_MAX      = 20    # fixed episode length (pad / truncate)
N_ACTIONS  = 25    # 5×5 action grid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HeatmapDataset(Dataset):
    """
    Converts each patient episode (variable-length time series) into a
    (3, 224, 224) image tensor built on the fly.

    Parameters
    ----------
    df           : DataFrame with normalised state columns, ID_COL,
                   TIME_COL, and ACTION_COL.
    patient_list : array of patient IDs to include.
    label_list   : integer action label for each patient (most-frequent action).
    state_cols   : list of feature column names (default STATE_COLS).
    t_max        : episode length after padding / truncation.
    """

    def __init__(
        self,
        df,
        patient_list: np.ndarray,
        label_list: np.ndarray,
        state_cols: list = None,
        t_max: int = T_MAX,
    ):
        self.df           = df
        self.patient_list = patient_list
        self.label_list   = label_list
        self.state_cols   = state_cols or STATE_COLS
        self.t_max        = t_max

    def __len__(self) -> int:
        return len(self.patient_list)

    def __getitem__(self, idx: int):
        pid   = self.patient_list[idx]
        group = self.df[self.df[ID_COL] == pid].sort_values(TIME_COL)
        mat   = group[self.state_cols].values.astype(np.float32)   # (T_i, n_features)

        # Pad or truncate to t_max rows
        n_feats = len(self.state_cols)
        if len(mat) > self.t_max:
            mat = mat[: self.t_max, :]
        else:
            pad = np.zeros((self.t_max - len(mat), n_feats), dtype=np.float32)
            mat = np.vstack([mat, pad])

        # Resize to 224×224 and replicate to 3 channels
        img = Image.fromarray((mat * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)   # (3, 224, 224)

        label = self.label_list[idx]
        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_imaging_splits(
    df,
    state_cols: list = None,
    test_size: float = 0.20,
    val_size: float  = 0.20,   # fraction of train+val (i.e. 16 % of total)
    random_state: int = 42,
):
    """
    Normalise *df* globally (visualisation only), assign a per-patient label
    (most-frequent action_combined), and return train / val / test patient arrays.

    Parameters
    ----------
    df            : preprocessed DataFrame containing state_cols + ID_COL
                    + TIME_COL + ACTION_COL
    state_cols    : feature columns to normalise; defaults to STATE_COLS
    test_size     : fraction of patients for the test set (default 0.20)
    val_size      : fraction of remaining patients for validation (default 0.20,
                    giving 16 % of the total)
    random_state  : random seed

    Returns
    -------
    df_norm                  : DataFrame with normalised features
    (X_train_pat, y_train)   : training patients & labels
    (X_val_pat,   y_val)     : validation patients & labels
    (X_test_pat,  y_test)    : test patients & labels
    """
    cols    = state_cols or STATE_COLS
    df_norm = df.copy()

    # Global normalisation for heatmap rendering (does NOT affect tabular splits)
    scaler         = MinMaxScaler()
    df_norm[cols]  = scaler.fit_transform(df_norm[cols])

    # Per-patient most-frequent action as label
    patient_ids    = df_norm[ID_COL].unique()
    patient_labels = np.array([
        df_norm[df_norm[ID_COL] == pid][ACTION_COL].mode()[0]
        for pid in patient_ids
    ])

    X_patients = np.array(patient_ids)

    # Step 1: carve out test set
    X_tv, X_test_pat, y_tv, y_test_pat = train_test_split(
        X_patients, patient_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_labels,
    )
    # Step 2: carve out val from the remaining pool
    X_train_pat, X_val_pat, y_train_pat, y_val_pat = train_test_split(
        X_tv, y_tv,
        test_size=val_size,
        random_state=random_state,
        stratify=y_tv,
    )

    print(
        f"Imaging split — Train: {len(X_train_pat)} patients, "
        f"Val: {len(X_val_pat)}, Test: {len(X_test_pat)}"
    )
    return (
        df_norm,
        (X_train_pat, y_train_pat),
        (X_val_pat,   y_val_pat),
        (X_test_pat,  y_test_pat),
    )


def build_data_loaders(
    df_norm,
    train_split: tuple,
    val_split: tuple,
    test_split: tuple,
    batch_size: int = 32,
    state_cols: list = None,
    t_max: int = T_MAX,
):
    """Wrap patient splits in HeatmapDataset and return (train, val, test) DataLoaders."""
    cols = state_cols or STATE_COLS

    train_ds = HeatmapDataset(df_norm, *train_split, state_cols=cols, t_max=t_max)
    val_ds   = HeatmapDataset(df_norm, *val_split,   state_cols=cols, t_max=t_max)
    test_ds  = HeatmapDataset(df_norm, *test_split,  state_cols=cols, t_max=t_max)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_resnet18(n_actions: int = N_ACTIONS, dropout: float = 0.5) -> nn.Module:
    """
    Load pretrained ResNet-18, freeze the backbone, and replace the head with
    a two-layer MLP that outputs *n_actions* logits.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Linear(128, n_actions),
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_resnet18(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 20,
    lr: float = 1e-3,
    save_path: str = "best_resnet18_sepsis.pth",
    device: torch.device = None,
    return_logs: bool = False,
):
    """
    Train *model* for *n_epochs* epochs, saving the best checkpoint by
    validation accuracy.

    Parameters
    ----------
    model        : ResNet-18 with custom head (from build_resnet18)
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    n_epochs     : number of training epochs
    lr           : learning rate for Adam (head parameters only)
    save_path    : file path for the best checkpoint (.pth)
    device       : torch device; auto-detected if None
    return_logs  : if True, return (model, epoch_logs) instead of just model
                   epoch_logs is a list of dicts {epoch, loss, val_acc}

    Returns
    -------
    model  (with best validation-accuracy weights loaded)
    — or —
    (model, epoch_logs)  when return_logs=True
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    best_val_acc = 0.0
    logs         = []

    for epoch in range(n_epochs):
        # ---- training ----
        model.train()
        total_loss = 0.0
        n_samples  = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            n_samples  += images.size(0)

        # ---- validation ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, pred = torch.max(model(images), 1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)

        val_acc    = correct / total
        epoch_loss = total_loss / n_samples
        logs.append({"epoch": epoch + 1, "loss": epoch_loss, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch {epoch + 1:2d}/{n_epochs} | "
            f"Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        gc.collect()

    # Restore best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nBest val accuracy: {best_val_acc:.4f} — checkpoint: {save_path}")

    if return_logs:
        return model, logs
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_resnet18(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = None,
) -> dict:
    """
    Run inference on *test_loader* and return accuracy + macro F1.

    Returns
    -------
    dict with keys: model_name, accuracy, macro_f1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            _, pred = torch.max(model(images), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"{'ResNet-18 (frozen)':30s} | Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    return {"model_name": "ResNet-18 (frozen)", "accuracy": acc, "macro_f1": f1}