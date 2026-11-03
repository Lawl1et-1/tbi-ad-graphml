"""Check our ComBat on simulated data with known batch shifts and scales."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from combat import combat  # noqa: E402


def simulate(seed=0, n_genes=300, sizes=(40, 60, 80)):
    rng = np.random.default_rng(seed)
    base = rng.normal(8, 2, n_genes)
    rows, batch, group = [], [], []
    for b, n in enumerate(sizes):
        shift = rng.normal(0, 1.5, n_genes)
        scale = rng.uniform(0.3, 3.0, n_genes) * (b + 1)  # very different scales per batch
        g = rng.integers(0, 2, n)
        signal = np.outer(g, np.r_[np.full(20, 1.0), np.zeros(n_genes - 20)])  # 20 true DE genes
        rows.append(base + shift + signal + rng.normal(0, 0.5, (n, n_genes)) * scale)
        batch += [b] * n
        group += list(g)
    X = pd.DataFrame(np.vstack(rows))
    return X, np.array(batch), np.array(group)


def test_removes_batch_location_and_scale():
    X, batch, _ = simulate()
    Y = combat(X, batch)
    means = Y.groupby(batch).mean()
    sds = Y.groupby(batch).std()
    assert (means.max() - means.min()).median() < 0.15  # batch means aligned
    ratio = (sds.max() / sds.min()).median()
    assert ratio < 1.3, ratio  # batch spreads aligned
    # Output stays on the original scale (unlike the PyPI package).
    assert 0.3 < Y.var().median() / X.var().median() * 1 < 3


def test_keeps_signal_with_balanced_design():
    X, batch, group = simulate(seed=1)
    Y = combat(X, batch)
    diff = Y[group == 1].mean() - Y[group == 0].mean()
    assert diff[:20].mean() > 5 * diff[20:].abs().mean()


if __name__ == "__main__":
    test_removes_batch_location_and_scale()
    test_keeps_signal_with_balanced_design()
    print("combat tests passed")
