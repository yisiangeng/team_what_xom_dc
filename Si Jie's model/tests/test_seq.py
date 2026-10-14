"""M13 to M16: dwc26.models.seq (SeqNet, train_member, SequenceProxy)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dwc26.data.check import ContractError  # noqa: E402
from dwc26.evaluation.metrics import objective_j  # noqa: E402
from dwc26.models import seq  # noqa: E402
from dwc26.models.lstm import LSTMProxy  # noqa: E402
from dwc26.models.rnn import RNNProxy  # noqa: E402
from dwc26.models.seq import SeqNet, SequenceProxy, time_grid, train_member  # noqa: E402
from synth_models import curves_data, fit_settings  # noqa: E402

P = 4


@pytest.mark.parametrize("cell", ["lstm", "gru", "rnn"])
@pytest.mark.parametrize("option, size", [("A", P + 1), ("B", 1), ("C", P)])
def test_m13_input_options(cell, option, size):
    torch.manual_seed(0)
    net = SeqNet(cell, option, P, num_layers=2, hidden_size=16)
    assert net.rnn.input_size == size
    u = torch.rand(5, P) * 2 - 1
    t = torch.linspace(0.1, 1.0, 7)
    z = net.inputs(u, t)
    assert z.shape == (5, 7, size)
    s0 = net.initial_state(u)
    if option == "B":
        h0 = s0[0] if cell == "lstm" else s0
        assert h0.shape == (2, 5, 16) and (h0.abs() < 1).all()
        if cell == "lstm":
            assert s0[1].shape == (2, 5, 16)
    else:
        assert s0 is None
    if option == "C":
        assert all(torch.equal(z[:, k], z[:, 0]) for k in range(7))    # constant input over the steps
        assert torch.equal(z[:, 0], u)
    if option == "A":
        assert torch.equal(z[:, 3, :P], u) and torch.allclose(z[:, 3, P], t[3].expand(5))
    y = net(u, t)
    assert y.shape == (5, 7, 3)
    u2 = u.clone()
    u2[:, 1] += 0.3
    assert not torch.allclose(net(u2, t), y)                           # a change in X changes the output


def test_m14_time_scaling(tmp_path, monkeypatch):
    seen = []
    orig = SeqNet.forward

    def spy(self, u, t):
        seen.append(t.detach().double().numpy().copy())
        return orig(self, u, t)

    monkeypatch.setattr(SeqNet, "forward", spy)
    X, Y, Xv, Yv = curves_data()
    m = LSTMProxy().fit(X, Y, fit_settings("lstm", tmp_path), val=(Xv, Yv))
    want = np.arange(2, 13) / 12.0
    assert len(seen) > 0 and all(np.allclose(t, want, rtol=0, atol=1e-7) for t in seen)
    np.testing.assert_array_equal(time_grid(2, 13, 13), want)
    seen.clear()
    m.rollout(Xv, 20)
    assert all(np.allclose(t, np.arange(2, 20) / 12.0, atol=1e-7) for t in seen) and seen[0].max() > 1
    seen.clear()
    LSTMProxy().fit(X, Y[:, :9], fit_settings("lstm", tmp_path), val=(Xv, Yv[:, :9]))
    assert all(np.allclose(t, np.arange(2, 9) / 8.0, atol=1e-7) for t in seen)


@pytest.mark.parametrize("option", ["A", "B", "C"])
def test_m15_early_stopping(tmp_path, option):
    X, Y, Xv, Yv = curves_data()
    s = fit_settings("lstm", tmp_path, variant=option)
    s["max_epochs"], s["patience"] = 5, 2
    r = train_member(s, 0, X, Y, Xv, Yv, 2)
    h = r.history
    assert h.shape == (r.n_epochs, 3) and np.array_equal(h[:, 0], np.arange(1, r.n_epochs + 1))
    assert r.best_epoch == int(h[np.argmin(h[:, 2]), 0])
    assert r.j_val == h[:, 2].min()
    assert r.n_epochs <= r.best_epoch + s["patience"]
    m = LSTMProxy.from_members(option, s, [r], 2, 13, P)
    assert objective_j(Yv, m.predict(Xv), (0, 2)) == pytest.approx(r.j_val, abs=1e-12)
    assert r.fit_s > 0 and r.seed == 0
    assert all(v.device.type == "cpu" for v in r.state.values())


def test_m15_patience_stops(tmp_path):
    X, Y, Xv, Yv = curves_data()
    s = fit_settings("lstm", tmp_path)
    s["max_epochs"], s["patience"], s["lr"] = 60, 1, 1.0e-2
    r = train_member(s, 0, X, Y, Xv, Yv, 2)
    assert r.n_epochs <= r.best_epoch + 1


def test_m16_ensemble(tmp_path):
    X, Y, Xv, Yv = curves_data()
    s = fit_settings("lstm", tmp_path, variant="B")
    members = [train_member(s, sd, X, Y, Xv, Yv, 2) for sd in s["seeds"]]
    m = LSTMProxy.from_members("B", s, members, 2, 13, P)
    assert m.variant == "B" and m.k0 == 2 and m.T_fit == 13
    single = [LSTMProxy.from_members("B", s, [r], 2, 13, P) for r in members]
    G = []
    for one in single:
        one_raw = one._raw_rollout(Xv, 13)
        G.append(one_raw)
    G = np.stack(G)
    mu = G.mean(axis=0)
    P_ = m.predict(Xv)
    np.testing.assert_allclose(P_[:, 2:], np.maximum(mu, 0.0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.predict_std(Xv)[:, 2:], G.std(axis=0, ddof=1), rtol=0, atol=1e-12)
    fitted = LSTMProxy().fit(X, Y, s, val=(Xv, Yv))
    assert np.array_equal(fitted.predict(Xv), P_)
    assert np.array_equal(fitted.predict_std(Xv), m.predict_std(Xv))


def test_cell_checks(tmp_path):
    X, Y, Xv, Yv = curves_data()
    s = fit_settings("lstm", tmp_path)
    with pytest.raises(ContractError):
        RNNProxy().fit(X, Y, s, val=(Xv, Yv))                       # cell lstm in an RNNProxy
    s = fit_settings("rnn", tmp_path, cell="gru")
    with pytest.raises(ContractError):
        LSTMProxy().fit(X, Y, s, val=(Xv, Yv))
    m = RNNProxy().fit(X, Y, s, val=(Xv, Yv))
    assert isinstance(m.nets[0].rnn, torch.nn.GRU)
    s1 = fit_settings("lstm", tmp_path)
    one = LSTMProxy.from_members("A", s1, [train_member(s1, 0, X, Y, Xv, Yv, 2)], 2, 13, P)
    assert isinstance(one, SequenceProxy)
    with pytest.raises(ContractError):
        one.predict_std(Xv)                                           # std needs >= 2 members


def test_nan_loss_raises_trial_failed(tmp_path):
    from dwc26.models.base import TrialFailed

    X, Y, Xv, Yv = curves_data()
    s = fit_settings("lstm", tmp_path)
    Ybad = Y.copy()
    Ybad[0, 5, 0] = 1e30                                              # float32 overflow in the loss
    with pytest.raises(TrialFailed):
        train_member(s, 0, X, Ybad * 1e10, Xv, Yv, 2)


def test_predict_batch_chunks(tmp_path):
    X, Y, Xv, Yv = curves_data()
    s = fit_settings("lstm", tmp_path)
    m = LSTMProxy().fit(X, Y, s, val=(Xv, Yv))
    Xq = np.random.default_rng(0).random((600, 4))
    a = m.predict(Xq)
    m.settings = dict(m.settings, predict_batch=256)
    b = m.predict(Xq)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    assert seq.DTYPES["float32"] is torch.float32
