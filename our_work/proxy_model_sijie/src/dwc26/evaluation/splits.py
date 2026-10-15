"""The arrays of one split, and the train reference statistics (spec task3 section 1.3).

Rows are selected by the stored `split` column (V2, V5). A `test` split is built
only with a running test-scoring ticket (V3, guards G1 to G7): the ticket is
verified before any test row is selected. The returned object holds no row of
another split.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from our_work.proxy_model_sijie.src.dwc26.data.check import ContractError, assert_array
from our_work.proxy_model_sijie.src.dwc26.data.db import SPLITS, read_task1_arrays
from our_work.proxy_model_sijie.src.dwc26.data.load import FLUIDS
from our_work.proxy_model_sijie.src.dwc26.data.prepare import arrays_sha256
from our_work.proxy_model_sijie.src.dwc26.evaluation.guards import TestTicket, verify_ticket
from our_work.proxy_model_sijie.src.dwc26.evaluation.plausibility import gor_range
from our_work.proxy_model_sijie.src.dwc26.models.base import onset_index
from our_work.proxy_model_sijie.src.dwc26.models.dataset import _stored_sha


@dataclass(frozen=True)
class SplitArrays:
    split: str                 # 'train' | 'val' | 'test'
    case_id: np.ndarray        # (n,) int64, ascending
    X: np.ndarray              # (n, p), scaled, [0, 1]
    Y_phys: np.ndarray         # (n, T, 3), physical, >= 0
    dates: np.ndarray          # (T,) datetime64[D]
    t_days: np.ndarray         # (T,)
    hist_phys: np.ndarray      # (T, 3), R1 history rates (D-004)
    y_factor: np.ndarray       # (3,)
    prepare_run_id: str
    arrays_sha256: str

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def T(self) -> int:
        return int(self.Y_phys.shape[1])

    @property
    def Y_scaled(self) -> np.ndarray:
        """Y_phys / y_factor: bit-equal to the Task 1 scaled Y (YScaler.transform)."""
        return self.Y_phys / self.y_factor


@dataclass(frozen=True)
class TrainRef:                # statistics of the train cases only (V1)
    k0: int
    mean_curve: np.ndarray     # (T, 3) physical, the B0 curve
    gor_range: tuple[float, float]


def load_split(con: sqlite3.Connection, split: str, *, ticket: TestTicket | None = None) -> SplitArrays:
    """Steps 1 to 4 of spec section 1.3. For 'test', the ticket is verified first (GuardError)."""
    if split not in SPLITS:
        raise ContractError(f"load_split: unknown split {split!r}; expected one of {list(SPLITS)}")
    if split == "test":
        verify_ticket(con, ticket)                     # GuardError before any test row is selected
    elif ticket is not None:
        raise ContractError(f"load_split: a test ticket is only for the test split, not {split!r}")
    a = read_task1_arrays(con)                         # F2 guard
    run_id = str(a["run_id"])
    sha = arrays_sha256(a)
    stored = _stored_sha(con, run_id)
    if sha != stored:
        raise ContractError(f"load_split: arrays_sha256 of the Task 1 tables ({sha[:12]}...) differs from "
                            f"runs.outputs.arrays_sha256 of prepare run {run_id} ({str(stored)[:12]}...); "
                            "rerun prepare")
    sel = a["split"] == split                          # stored split column (V2, V5)
    case_id = a["case_id"][sel].astype(np.int64)
    X = a["X"][sel].copy()
    Y_phys = a["Y_phys"][sel].copy()
    dates, t_days = a["dates"].copy(), a["t_days"].copy()
    hist_phys, y_factor = a["hist_phys"].copy(), a["y_factor"].copy()
    fluids = tuple(str(s) for s in a["fluids"])
    del a, sel                                         # no row of another split is kept

    n, T = len(case_id), len(dates)
    if n < 1:
        raise ContractError(f"load_split: no case with split {split!r}")
    if fluids != FLUIDS:
        raise ContractError(f"load_split: fluids {fluids}, expected {FLUIDS}")
    assert_array(f"load_split {split} X", X, (n, None))
    if (X < 0).any() or (X > 1).any():
        raise ContractError(f"load_split {split} X: values outside [0, 1]")
    assert_array(f"load_split {split} Y_phys", Y_phys, (n, T, len(FLUIDS)), nonneg=True)
    assert_array("load_split hist_phys", hist_phys, (T, len(FLUIDS)), nonneg=True)
    assert_array("load_split dates", dates, (T,), dtype="datetime64[D]")
    assert_array("load_split t_days", t_days, (T,))
    assert_array("load_split y_factor", y_factor, (len(FLUIDS),))
    if not (y_factor > 0).all():
        raise ContractError("load_split: y_factor must be > 0")
    if not (np.diff(case_id) > 0).all():
        raise ContractError("load_split: case IDs are not strictly ascending")
    if not (np.diff(dates.astype(np.int64)) > 0).all():
        raise ContractError("load_split: dates are not strictly increasing")
    for arr in (case_id, X, Y_phys, dates, t_days, hist_phys, y_factor):
        arr.setflags(write=False)
    return SplitArrays(split=split, case_id=case_id, X=X, Y_phys=Y_phys, dates=dates, t_days=t_days,
                       hist_phys=hist_phys, y_factor=y_factor, prepare_run_id=run_id, arrays_sha256=sha)


def train_reference(train: SplitArrays) -> TrainRef:
    """k0, the mean training curve (B0) and the GOR range, from the train cases only (V1)."""
    if train.split != "train":
        raise ContractError(f"train_reference: needs the train split, got {train.split!r}")
    k0 = onset_index(train.Y_scaled)
    mean_curve = train.Y_phys.mean(axis=0)
    mean_curve.setflags(write=False)
    return TrainRef(k0=k0, mean_curve=mean_curve, gor_range=gor_range(train.Y_phys))
