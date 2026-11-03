"""Phase 5 design selection for the GAT, on validation data only.

Uses the TRAINING part of CV fold 0 (seed 42) and splits it again by donor (80/20). Test folds are
never touched, so the later CV estimate stays unbiased. Output: results/phase5/dev_gat_variants.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gat import GATClassifier, build_edge_index  # noqa: E402
from phase5_models import OUT, PROC, load, splits, zscore  # noqa: E402

torch.set_num_threads(8)
BASE = dict(hidden=8, heads=2, emb_dim=4, batch_size=16, epochs=40, patience=8, weight_decay=1e-4)
VARIANTS = {
    "A_pool_lr1e-3_min15": dict(readout="pool", lr=1e-3, min_epochs=15),
    "B_gene_lr1e-3_min15": dict(readout="gene", lr=1e-3, min_epochs=15),
    "B_gene_lr3e-3_min10": dict(readout="gene", lr=3e-3, min_epochs=10),
}


def main() -> None:
    X, meta, y = load()
    sp = splits(meta, y)
    tr = (sp["fold_s42"] != 0).values
    Xtr, ytr, gtr = X.values[tr].astype(np.float32), y[tr], meta.donor_id.values[tr]
    a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=7).split(Xtr, ytr, gtr))
    Za, Zb = zscore(Xtr[a], Xtr[b])
    # Early-stopping split carved from the dev-train part only.
    a2, a3 = next(GroupShuffleSplit(1, test_size=0.15, random_state=8).split(Za, ytr[a], gtr[a]))
    genes = (PROC / "selected_genes.txt").read_text().split("\n")
    ei = build_edge_index(pd.read_csv(PROC / "graph_string.csv"), genes)
    rows = []
    for name, v in VARIANTS.items():
        t0 = time.time()
        m = GATClassifier(ei, len(genes), seed=42, **BASE, **v).fit(Za[a2], ytr[a][a2], Za[a3], ytr[a][a3])
        auc = roc_auc_score(ytr[b], m.predict_proba(Zb)[:, 1])
        rows.append({"variant": name, "dev_val_auroc": auc, "best_epoch": m.best_epoch,
                     "epochs_run": len(m.history), "seconds": round(time.time() - t0)})
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT / "dev_gat_variants.csv", index=False)


if __name__ == "__main__":
    main()
