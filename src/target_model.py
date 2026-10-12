"""PyTorch mechanistic disease classifier integrating molecular + clinical
data, plus permutation-importance-based target identification (ranking
candidate genes by how much held-out accuracy degrades when each gene's
values are shuffled -- a real, model-agnostic importance measure)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .simulate_data import GENES, CLINICAL

FEATURES = GENES + CLINICAL
SEED = 97


class MechanisticMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_evaluate(df, seed: int = SEED, epochs: int = 150, lr: float = 0.01) -> dict:
    torch.manual_seed(seed)
    X = df[FEATURES].values.astype("float32")
    y = df["disease"].values.astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = MechanisticMLP(n_features=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(torch.tensor(X_test))).numpy()
    preds = (test_probs > 0.5).astype(int)
    accuracy = round(float(accuracy_score(y_test, preds)), 4)

    return {"model": model, "accuracy": accuracy, "X_test": X_test, "y_test": y_test,
            "n_train": len(X_train), "n_test": len(X_test)}


def permutation_importance(model, X_test: np.ndarray, y_test: np.ndarray, seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        base_preds = (torch.sigmoid(model(torch.tensor(X_test))).numpy() > 0.5).astype(int)
    base_acc = accuracy_score(y_test, base_preds)

    results = []
    for i, feature in enumerate(FEATURES):
        X_perm = X_test.copy()
        rng.shuffle(X_perm[:, i])
        with torch.no_grad():
            perm_preds = (torch.sigmoid(model(torch.tensor(X_perm))).numpy() > 0.5).astype(int)
        perm_acc = accuracy_score(y_test, perm_preds)
        results.append({"feature": feature, "accuracy_drop": round(float(base_acc - perm_acc), 4)})

    results.sort(key=lambda r: -r["accuracy_drop"])
    return results


if __name__ == "__main__":
    from .simulate_data import simulate

    df = simulate()
    result = train_and_evaluate(df)
    print("Accuracy:", result["accuracy"])
    for r in permutation_importance(result["model"], result["X_test"], result["y_test"]):
        print(r)
