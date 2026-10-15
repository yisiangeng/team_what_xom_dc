"""E13: dwc26.forecast.dates (spec task3 section 1.3)."""

from __future__ import annotations

import numpy as np
import pytest

from dwc26.data.check import ContractError
from dwc26.forecast.dates import forecast_dates, quarters_between


def test_e13_quarters_between():
    assert quarters_between(np.datetime64("2008-01-01"), np.datetime64("2028-01-01")) == 80
    assert quarters_between(np.datetime64("2003-01-01"), np.datetime64("2005-01-01")) == 8
    assert quarters_between(np.datetime64("2005-01-01"), np.datetime64("2005-01-01")) == 0
    with pytest.raises(ContractError):
        quarters_between(np.datetime64("2008-01-01"), np.datetime64("2028-02-01"))
    with pytest.raises(ContractError):
        quarters_between(np.datetime64("2008-01-01"), np.datetime64("2028-01-02"))


def test_e13_forecast_dates():
    got = forecast_dates(np.datetime64("2008-01-01"), 3)
    assert got.dtype == np.dtype("datetime64[D]")
    assert got.tolist() == np.array(["2008-04-01", "2008-07-01", "2008-10-01"], dtype="datetime64[D]").tolist()
    eighty = forecast_dates(np.datetime64("2008-01-01"), 80)
    assert eighty.shape == (80,) and str(eighty[-1]) == "2028-01-01"
    assert forecast_dates(np.datetime64("2008-01-01"), 0).shape == (0,)
