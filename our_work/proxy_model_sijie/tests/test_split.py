"""A23: dwc26.data.split."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.config import ConfigError
from dwc26.data.split import assign_split

DEFAULT = {"train": [1, 70], "val": [71, 85], "test": [86, 100]}


def test_a23_default_split():
    ids = np.arange(1, 101, dtype=np.int64)
    s = assign_split(ids, DEFAULT)
    assert s.dtype == np.dtype("<U5") and s.shape == (100,)
    assert (s == "train").sum() == 70
    assert (s == "val").sum() == 15
    assert (s == "test").sum() == 15
    assert (s[:70] == "train").all() and (s[70:85] == "val").all() and (s[85:] == "test").all()


def test_split_by_case_number_not_position():
    ids = np.array([86, 1, 71], dtype=np.int64)
    np.testing.assert_array_equal(assign_split(ids, DEFAULT), ["test", "train", "val"])


@pytest.mark.parametrize(
    "ranges",
    [
        {"train": [1, 69], "val": [71, 85], "test": [86, 100]},   # case 70 in no range
        {"train": [1, 71], "val": [71, 85], "test": [86, 100]},   # case 71 in two ranges
        {"train": [1, 70], "val": [71, 85], "test": [86, 99]},    # case 100 in no range
        {"train": [70, 1], "val": [71, 85], "test": [86, 100]},   # inverted
        {"train": [1, 70], "val": [71, 85]},                      # missing test
        {"train": [1, 70], "val": [71, 85], "test": [86]},        # malformed
        {"train": [1, 70], "val": [71, 85], "test": [86, 100], "extra": [101, 102]},
    ],
)
def test_a23_invalid_split(ranges):
    ids = np.arange(1, 101, dtype=np.int64)
    with pytest.raises(ConfigError):
        assign_split(ids, ranges)
