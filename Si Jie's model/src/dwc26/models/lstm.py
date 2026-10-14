"""LSTM proxy: nn.LSTM (spec task2 section 2.6; PRD section 6; Article 2)."""

from __future__ import annotations

from dwc26.models.seq import SequenceProxy


class LSTMProxy(SequenceProxy):
    name = "lstm"
    cells = ("lstm",)
