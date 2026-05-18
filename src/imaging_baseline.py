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


STATE_COLS = [
    "HR", "SysBP", "MeanBP", "DiaBP", "RR", "Temp", "SpO2", "GCS",
    "pH", "BE", "HCO3", "FiO2", "PaO2", "PaCO2", "Lactate",
    "K", "Na", "Cl", "Ca", "IonisedCa", "CO2", "Mg",
    "BUN", "Creatinine", "Albumin", "SGOT", "SGPT", "TotalBili",
    "Hb", "WbcCount", "PlateletsCount", "PTT", "PT", "INR", "Age",
]

ID_COL     = "PatientID"
TIME_COL   = "Timepoints"
ACTION_COL = "action_combined"

T_MAX     = 20
N_ACTIONS = 25


class HeatmapDataset(Dataset):
    """
    Converts each patient episode into a (3, 224, 224) image tensor.

    The heatmap is built from heatmap_cols (base physiological features only).
    Labels come from ACTION_COL (action_combined), which must be present in df.
    """

    def __init__(self, df, patient_list, label_list, heatmap_cols=None, t_max=T_MAX):
        self.df           = df
        self.patient_list = patient_list
        self.label_list   = label_list
        self.heatmap_cols = heatmap_cols or STATE_COLS
        self.t_max        = t_max

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        pid   = self.patient_list[idx]
        group = self.df[self.df[ID_COL] == pid].sort_values(TIME_COL)
        mat   = group[self.heatmap_cols].values.astype(np.float32)

        n_feats = len(self.heatmap_cols)
        if len(mat) > self.t_max:
            mat = mat[: self.t_max, :]
        else:
            pad = np.zeros((self.t_max - len(mat), n_feats), dtype=np.float32)
            mat = np.vstack([mat, pad])

        img = Image.fromarray((mat * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.stack([img, img, img], axis=0)

        return torch.from_numpy(img), torch.tensor(self.label_list[idx], dtype=torch.long)


def prepare_imaging_splits(
    df,
    state_cols=None,
    test_size=0.20,
    val_size=0.20,
    random_state=42,
):
    """
    Prepare patient-level train/val/test splits for the imaging baseline.

    Parameters
    ----------
    df          : the fully preprocessed DataFrame (sepsis_preprocessed.csv).
                  Must contain ID_COL, TIME_COL, ACTION_COL, and state_cols.
                  Only state_cols columns are used for the heatmap pixels;
                  ACTION_COL is used only for patient-level labels.
    state_cols  : base physiological columns used to build the heatmap
                  (default STATE_COLS — 35 columns, no engineered features).
    test_size   : fraction of patients for the test set (default 0.20).
    val_size    : fraction of remaining patients for validation (default 0.20,
                  giving 16 % of the total).
    random_state: random seed.

    Returns
    -------
    df_norm              : DataFrame with state_cols normalised to [0, 1].
    (X_train_pat, y_tr)  : training patient IDs and labels.
    (X_val_pat,   y_val) : validation patient IDs and labels.
    (X_test_pat,  y_te)  : test patient IDs and labels.
    """
    cols = state_cols or STATE_COLS

    assert ACTION_COL in df.columns, (
        f"'{ACTION_COL}' not found in DataFrame. "
        "Pass sepsis_preprocessed.csv (after feature engineering), not sepsis_clean.csv."
    )
    for c in cols:
        assert c in df.columns, f"Heatmap column '{c}' not found in DataFrame."

    df_norm = df.copy()
    scaler  = MinMaxScaler()
    df_norm[cols] = scaler.fit_transform(df_norm[cols])

    patient_ids    = df_norm[ID_COL].unique()
    patient_labels = np.array([
        df_norm[df_norm[ID_COL] == pid][ACTION_COL].mode()[0]
        for pid in patient_ids
    ])

    X_tv, X_test_pat, y_tv, y_test_pat = train_test_split(
        patient_ids, patient_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_labels,
    )
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


def build_data_loaders(df_norm, train_split, val_split, test_split,
                       batch_size=32, state_cols=None, t_max=T_MAX):
    """Wrap patient splits in HeatmapDataset and return (train, val, test) DataLoaders."""
    cols = state_cols or STATE_COLS

    train_ds = HeatmapDataset(df_norm, *train_split, heatmap_cols=cols, t_max=t_max)
    val_ds   = HeatmapDataset(df_norm, *val_split,   heatmap_cols=cols, t_max=t_max)
    test_ds  = HeatmapDataset(df_norm, *test_split,  heatmap_cols=cols, t_max=t_max)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0),
    )


def build_resnet18(n_actions=N_ACTIONS, dropout=0.5):
    """Load pretrained ResNet-18, freeze backbone, replace head with a 2-layer MLP."""
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


def train_resnet18(
    model, train_loader, val_loader,
    n_epochs=20, lr=1e-3,
    save_path="best_resnet18_sepsis.pth",
    device=None,
    return_logs=False,
):
    """
    Train model for n_epochs epochs, saving the best checkpoint by validation accuracy.

    Parameters
    ----------
    return_logs : if True, return (model, epoch_logs) where epoch_logs is a list of
                  dicts {epoch, loss, val_acc}. If False, return model only.
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
        model.train()
        total_loss, n_samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            n_samples  += images.size(0)

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

        print(f"Epoch {epoch+1:2d}/{n_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")
        gc.collect()

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nBest val accuracy: {best_val_acc:.4f} — checkpoint: {save_path}")

    return (model, logs) if return_logs else model


def evaluate_resnet18(model, test_loader, device=None):
    """Run inference on test_loader and return accuracy + macro F1."""
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