from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models.base_classifier import BaseClassifier


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int,
                 n_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMClassifier(BaseClassifier):

    def __init__(self, seq_len: int = 20, hidden_size: int = 64, n_layers: int = 2,
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._net: _LSTMNet | None = None
        self._le = LabelEncoder()
        self._scaler = StandardScaler()

    @property
    def name(self) -> str:
        return "lstm"

    @property
    def default_params(self) -> dict:
        return {
            "seq_len": 20, "hidden_size": 64, "n_layers": 2,
            "dropout": 0.2, "lr": 1e-3, "epochs": 30, "batch_size": 64,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_s = self._scaler.fit_transform(X)
        y_enc = self._le.fit_transform(y)
        n_classes = len(self._le.classes_)

        X_seq, y_seq = _make_sequences(X_s, y_enc, self.seq_len)
        if len(X_seq) == 0:
            raise ValueError(f"Need > {self.seq_len} samples, got {len(X)}")

        self._net = _LSTMNet(X_seq.shape[2], self.hidden_size, self.n_layers,
                             n_classes, self.dropout)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self._net(xb), yb).backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not trained — call fit() first")
        X_s = self._scaler.transform(X)
        X_seq, _ = _make_sequences(X_s, np.zeros(len(X_s), dtype=int), self.seq_len)
        if len(X_seq) == 0:
            n_classes = len(self._le.classes_)
            return np.full((len(X), n_classes), 1.0 / n_classes)
        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.tensor(X_seq, dtype=torch.float32))
            proba_seq = torch.softmax(logits, dim=1).numpy()
        n_classes = proba_seq.shape[1]
        out = np.full((len(X), n_classes), 1.0 / n_classes)
        out[self.seq_len:] = proba_seq
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not trained — call fit() first")
        proba = self.predict_proba(X)
        return self._le.inverse_transform(proba.argmax(axis=1))


def _make_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X) - seq_len
    if n <= 0:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0, dtype=int)
    X_seq = np.stack([X[i: i + seq_len] for i in range(n)])
    y_seq = y[seq_len:]
    return X_seq, y_seq
