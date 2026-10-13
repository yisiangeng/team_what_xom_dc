r"""Health check for proxy_outputs/final_model.pt.

Loads the checkpoint, prints its metadata, and verifies that it reproduces the
predictions stored in proxy_outputs/predictions.csv for a held-out test case.

Run:  .\.venv\Scripts\python.exe .\check_checkpoint.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import proxy_model as pm

CODE = Path(__file__).resolve().parent
CKPT = CODE / "proxy_outputs" / "final_model.pt"
DATA = CODE / "data_after_engineering"


def main():
    print(f"file: {CKPT}  ({CKPT.stat().st_size / 1e6:.2f} MB)")
    # weights_only=False is needed here: the checkpoint holds numpy arrays and the config
    # dict, not only tensors (same reason as predict_case). Only load our own checkpoints.
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)

    print("\n-- metadata --")
    print("keys          :", list(ck.keys()))
    print("config        :", ck["config"])
    print("env           :", ck["env"])
    print("ensemble      :", len(ck["states"]), "members, seeds", ck["seeds"])
    print("params        :", ck["params"])
    print("grid features :", ck["grid_features"])
    print("dates / t_days:", len(ck["dates"]), "/", len(ck["t_days"]))
    print("param min/max :", np.round(ck["parameter_min"], 4), np.round(ck["parameter_max"], 4))

    print("\n-- round-trip check: predict test case 86 from raw values --")
    cases = pd.read_csv(DATA / "cases.csv")
    grid_feat = pd.read_csv(DATA / "grid_case_features.csv")
    params = cases.loc[cases.case_id == 86, pm.PARAMS].to_numpy()[0]
    grid = grid_feat.loc[grid_feat.case_id == 86, pm.GRID_FEATURES].to_numpy()[0]
    pred = pm.predict_case(CKPT, params, grid)

    stored = pd.read_csv(CODE / "proxy_outputs" / "predictions.csv")
    worst = 0.0
    for phase in pm.PHASES:
        s = stored[(stored.case_id == 86) & (stored.phase == phase)].sort_values("t_days")
        diff = float(np.abs(pred[f"{phase}_rate"].to_numpy() - s.rate_pred.to_numpy()).max())
        worst = max(worst, diff)
        print(f"  {phase:<5} max |checkpoint - stored| = {diff:.2e}")
    print("OK: the checkpoint reproduces the stored predictions" if worst < 0.01
          else "MISMATCH: checkpoint and predictions.csv disagree")


if __name__ == "__main__":
    main()
