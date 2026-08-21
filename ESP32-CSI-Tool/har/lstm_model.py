"""LSTM activity classifier, following Abuhoureyah et al. (2024) Section 4.1.4.

The paper's Fig. 13 stack is: input sequence -> LSTM -> dropout -> a second
LSTM stage -> dropout -> fully connected -> softmax over the activity classes.
That is what ``ActivityLSTM`` implements. The paper's own figure labels one
middle stage "Fully Connected LSTM Layer 2", which is not a standard layer
type; we read it as a second recurrent layer, which is the only reading
consistent with the surrounding text ("low-layering structure of the LSTM").

Requires PyTorch, so this module lives in ``lstm_env`` (Python 3.12), not the
3.14 ``csi_env``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-8


def relative_amplitude(windows: np.ndarray) -> np.ndarray:
    """Gain-invariant normalisation that PRESERVES relative variance.

    ``(x - mean_t) / mean_t`` per subcarrier per window. Dividing by each
    subcarrier's own mean removes transmitter distance and gain, which differ
    between rooms, while leaving the fluctuation magnitude intact.

    Per-window z-scoring is the obvious alternative and is wrong here: it
    rescales every window to unit variance, destroying exactly the difference
    between a still person and a walking one.
    """
    windows = np.asarray(windows, dtype=np.float32)
    mean = windows.mean(axis=1, keepdims=True)
    return (windows - mean) / (np.abs(mean) + EPS)


class ActivityLSTM(nn.Module):
    def __init__(self, n_subcarriers: int, n_classes: int, hidden: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(n_subcarriers, hidden, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):                       # x: [B, T, subcarriers]
        h, _ = self.lstm1(x)
        h = self.drop1(h)
        h, _ = self.lstm2(h)
        h = self.drop2(h[:, -1, :])             # last timestep
        return self.fc(h)                       # logits [B, n_classes]


class LSTMClassifier:
    """Inference wrapper with the same surface the demo expects.

    ``classes_`` and ``predict_proba`` mirror the scikit-learn API so
    live_demo.py can treat a sequence model and a feature model alike.
    """

    def __init__(self, net: ActivityLSTM, classes, scale: float, device: str = "cpu"):
        self.net = net.eval()
        self.classes_ = np.asarray(classes)
        self.scale = float(scale)
        self.device = device

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """windows: (n, frames, subcarriers) raw amplitudes -> (n, n_classes)."""
        x = relative_amplitude(windows) / (self.scale + EPS)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(x).float().to(self.device))
            return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(windows).argmax(axis=1)]


def train_lstm(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    classes: list,
    epochs: int = 60,
    hidden: int = 64,
    dropout: float = 0.3,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> LSTMClassifier:
    """Fit the classifier. The scale factor is derived from training data only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = relative_amplitude(train_windows)
    scale = float(x.std()) or 1.0
    x = x / scale
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[str(v)] for v in train_labels], dtype=np.int64)

    net = ActivityLSTM(x.shape[2], len(classes), hidden, dropout).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)

    # Class weights keep an uneven number of takes per label from biasing the
    # model toward whichever activity was recorded most.
    counts = np.bincount(y, minlength=len(classes)).astype(np.float32)
    weights = torch.tensor(len(y) / (len(classes) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    xt = torch.from_numpy(x).float().to(device)
    yt = torch.from_numpy(y).to(device)

    net.train()
    for epoch in range(epochs):
        order = torch.randperm(len(xt), device=device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimiser.zero_grad()
            loss = criterion(net(xt[batch]), yt[batch])
            loss.backward()
            # CSI windows produce occasional large gradients; clipping keeps a
            # short training run from diverging on one bad batch.
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimiser.step()
            total += float(loss.detach()) * len(batch)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1:3d}  loss {total / len(order):.4f}")

    return LSTMClassifier(net, classes, scale, device)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
