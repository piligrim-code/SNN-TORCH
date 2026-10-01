"""Tested helpers for the notebook's single-step LIF classifier."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import snntorch as snn
from snntorch import surrogate
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class CustomDataset(Dataset):
    def __init__(self, df, is_train=True, scaler=None, transform=None):
        self.transform = transform
        if not is_train and "class" in df.columns:
            raise ValueError("Remove the target column for unlabelled inference")
        self.labels = df["class"].to_numpy() if is_train else None
        frame = df.drop(columns=["class"]) if is_train else df
        self.features = np.asarray(scaler.transform(frame) if scaler is not None else frame, dtype=np.float32)
        if self.features.ndim != 2 or not all(self.features.shape) or not np.isfinite(self.features).all():
            raise ValueError("Features must be a finite, nonempty numeric table")
        if self.labels is not None:
            if not np.issubdtype(self.labels.dtype, np.integer) or (self.labels < 0).any():
                raise ValueError("Encode target labels as nonnegative integers first")
            self.labels = self.labels.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        result = {"feature": torch.as_tensor(feature, dtype=torch.float32)}
        if self.labels is not None:
            result["label"] = torch.tensor(self.labels[index], dtype=torch.long)
        return result


def make_loaders(train_dataset, test_dataset, *, batch_size=64, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
    )


class Net(nn.Module):
    """Two LIF layers reset for each forward call; no temporal unroll."""
    def __init__(self, n_features=10, n_classes=3, hidden=10, beta=0.5):
        super().__init__()
        if n_features < 1 or hidden < 1 or n_classes < 2:
            raise ValueError("Invalid model dimensions")
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(n_features, hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        spike1, _ = self.lif1(self.fc1(x), self.lif1.init_leaky())
        spike2, _ = self.lif2(self.fc2(spike1), self.lif2.init_leaky())
        return self.fc3(spike2)


def _batch(data, device):
    if "label" not in data:
        raise ValueError("Training/evaluation requires labels; use predict for inference")
    return data["feature"].to(device=device, dtype=torch.float32), data["label"].to(device=device, dtype=torch.long)


def _check_device(model, device):
    expected = torch.device(device)
    actual = next(model.parameters()).device
    if actual.type != expected.type or (expected.index is not None and actual.index != expected.index):
        raise ValueError("Move the model to the requested device before constructing its optimizer")
    return actual


def train_epoch(model, loader, optimizer, *, device="cpu"):
    device = _check_device(model, device)
    model.train()
    total_loss = 0.0
    seen = 0
    for data in loader:
        inputs, labels = _batch(data, device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        seen += len(labels)
    if not seen:
        raise ValueError("Empty training loader")
    return total_loss / seen


def classification_metrics(labels, probabilities, loss):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    cm = confusion_matrix(labels, predicted, labels=np.arange(classes))
    tp = np.diag(cm)
    fn, fp = cm.sum(axis=1) - tp, cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp

    def ratios(numerators, denominators):
        return [float(n / d) if d else None for n, d in zip(numerators, denominators)]

    sensitivity, specificity = ratios(tp, tp + fn), ratios(tn, tn + fp)
    def defined_mean(values):
        valid = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None

    auc = None
    if len(np.unique(labels)) == classes:
        auc = float(roc_auc_score(labels, probabilities[:, 1])) if classes == 2 else float(
            roc_auc_score(labels, probabilities, labels=np.arange(classes), multi_class="ovr"))
    return {"samples": len(labels), "loss": float(loss), "accuracy": float(np.mean(predicted == labels)),
            "confusion_matrix": cm.tolist(), "sensitivity": sensitivity, "specificity": specificity,
            "macro_sensitivity": defined_mean(sensitivity), "macro_specificity": defined_mean(specificity), "auc_roc": auc}


def evaluate(model, loader, *, device="cpu"):
    device = _check_device(model, device)
    if loader.drop_last:
        raise ValueError("Evaluation must not drop incomplete batches")
    model.eval()
    loss_sum, count = 0.0, 0
    all_labels, probabilities = [], []
    with torch.no_grad():
        for data in loader:
            inputs, labels = _batch(data, device)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
                raise ValueError("Expected finite class logits")
            loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            count += len(labels)
            all_labels.extend(labels.cpu().tolist())
            probabilities.extend(logits.softmax(dim=1).cpu().tolist())
    if not count or count != len(loader.dataset):
        raise ValueError("Evaluation must cover every sample in a nonempty dataset")
    return classification_metrics(all_labels, probabilities, loss_sum / count)


def predict(model, loader, *, device="cpu"):
    device = _check_device(model, device)
    if loader.drop_last:
        raise ValueError("Prediction must not drop incomplete batches")
    model.eval()
    result = []
    with torch.no_grad():
        for data in loader:
            logits = model(data["feature"].to(device=device, dtype=torch.float32))
            if logits.ndim != 2 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
                raise ValueError("Expected finite class logits")
            result.append(logits.softmax(dim=1).cpu())
    if not result or sum(len(batch) for batch in result) != len(loader.dataset):
        raise ValueError("Prediction must cover the complete nonempty dataset")
    return torch.cat(result).numpy()


def prepare_split(frame, *, test_size=0.2, seed=42, oversample=True):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["class"])
    train, test = train.copy(), test.copy()
    encoder = LabelEncoder().fit(train["class"])
    if len(encoder.classes_) < 2:
        raise ValueError("Classification needs at least two classes")
    train["class"] = encoder.transform(train["class"])
    test["class"] = encoder.transform(test["class"])
    scaler = StandardScaler().fit(train.drop(columns=["class"]))
    train_dataset = CustomDataset(train, scaler=scaler)
    test_dataset = CustomDataset(test, scaler=scaler)
    if oversample:
        from imblearn.over_sampling import SMOTE

        smallest = int(np.bincount(train_dataset.labels).min())
        if smallest < 2:
            raise ValueError("SMOTE requires at least two training examples per class")
        train_dataset.features, train_dataset.labels = SMOTE(
            random_state=seed, k_neighbors=min(5, smallest - 1)).fit_resample(
                train_dataset.features, train_dataset.labels)
    return train_dataset, test_dataset, scaler, encoder


def run_experiment(frame, *, epochs=20, lr=0.001, batch_size=64, seed=42, device="cpu", oversample=True):
    """Fixed training budgets; the test split is scored once per model, never tuned on."""
    if not isinstance(epochs, int) or epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("Need positive integer epochs and finite positive learning rate")
    train, test, scaler, encoder = prepare_split(frame, seed=seed, oversample=oversample)
    features, classes = train.features.shape[1], len(encoder.classes_)
    result = {"metrics": {}, "history": {}, "models": {}, "scaler": scaler, "encoder": encoder}
    for name in ("single_step_lif", "mlp_baseline"):
        set_seed(seed)
        model = Net(features, classes) if name == "single_step_lif" else nn.Sequential(
            nn.Linear(features, 10), nn.ReLU(), nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, classes))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader, test_loader = make_loaders(train, test, batch_size=batch_size, seed=seed)
        history = [train_epoch(model, train_loader, optimizer, device=device) for _ in range(epochs)]
        result["history"][name] = history
        result["metrics"][name] = evaluate(model, test_loader, device=device)
        result["models"][name] = model
    return result
