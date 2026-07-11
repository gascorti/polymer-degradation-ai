"""
Modelo LSTM (Long Short-Term Memory) para modelar la secuencia temporal de
pérdida de masa y predecir la constante cinética k (regresión).
HU3 del Product Backlog.

Nota: requiere `torch` instalado (ver requirements.txt).
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):
        def __init__(self, units):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=units, batch_first=True)
            self.fc1 = nn.Linear(units, 16)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h = torch.relu(self.fc1(h_n[-1]))
            return self.fc2(h).squeeze(-1)


class LSTMModel:
    """Wrapper con interfaz símil-Keras (`.predict`) sobre una red LSTM de PyTorch."""

    def __init__(self, net):
        self.net = net

    def predict(self, X, verbose=0):
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            y_pred = self.net(X_t).numpy()
        return y_pred.reshape(-1, 1)


def build_lstm_model(sequence_length: int, units: int = 32) -> LSTMModel:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch no está instalado. Ejecutá `pip install torch` para usar este modelo."
        )
    return LSTMModel(_LSTMNet(units))


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=60, batch_size=16, random_state=42):
    if not TORCH_AVAILABLE:
        raise ImportError("torch no está instalado.")

    torch.manual_seed(random_state)
    net = model.net
    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n = X_train_t.shape[0]
    patience = 10
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"loss": [], "val_loss": []}

    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X_val_t), y_val_t).item()

        history["loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return history
