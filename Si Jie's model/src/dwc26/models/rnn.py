"""RNN proxy: nn.RNN (tanh) or nn.GRU, the cell is a tuned setting (spec task2 section 2.6; PRD section 6)."""

from __future__ import annotations

from dwc26.models.seq import SequenceProxy


class RNNProxy(SequenceProxy):
    name = "rnn"
    cells = ("rnn", "gru")
