"""Tests for the history-matching / forecast pipeline (plan section 6).

Independent of the trained checkpoint except for the last class, which is skipped
when ``proxy_outputs/cumulative_model.pt`` does not exist yet. Run from ``cod/``:

    python -m unittest test_forecast_pipeline test_proxy_model
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import our_work.proxy_model_elton.forecast_pipeline as fp
import our_work.proxy_model_elton.proxy_model as proxy


class CumulativeDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fp.load_cumulative_data()

    def test_shapes_are_cumulative_targets(self):
        self.assertEqual(self.data["Y"].shape, (100, 41, 3))
        self.assertEqual(self.data["y_hist"].shape, (41, 3))
        # cumulative curves are non-decreasing in time for every case and phase
        self.assertTrue((np.diff(self.data["Y"], axis=1) >= -1e-6).all())
        self.assertTrue((np.diff(self.data["y_hist"], axis=0) >= -1e-6).all())

    def test_phase_axis_is_reordered_from_template_to_pipeline_order(self):
        # the npz stores [gas, oil, water]; the pipeline uses PHASES = [oil, gas, water]
        history = pd.read_csv(proxy.DATA / "history_observed.csv")
        for j, phase in enumerate(proxy.PHASES):
            np.testing.assert_allclose(self.data["y_hist"][:, j],
                                       history[f"{phase}_cum"].to_numpy(),
                                       rtol=1e-12, atol=1e-6)
        quarterly = pd.read_csv(proxy.DATA / "curves_quarterly.csv")
        case_one = quarterly[quarterly.case_id == 1]
        for j, phase in enumerate(proxy.PHASES):
            np.testing.assert_allclose(self.data["Y"][0, :, j],
                                       case_one[f"{phase}_cum"].to_numpy(),
                                       rtol=1e-12, atol=1e-6)


class GridMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fp.load_cumulative_data()

    def test_analytic_or_fallback_mapping_matches_all_100_stored_cases(self):
        mapper = fp.GridMapper(self.data)
        cases = self.data["cases"]
        theta = cases[proxy.PARAMS].to_numpy(dtype=np.float64)
        stored = self.data["grid"][proxy.GRID_FEATURES].to_numpy(dtype=np.float64)
        train = cases["split"].to_numpy() == "train"
        sigma = np.where(stored[train].std(axis=0) > 0, stored[train].std(axis=0), 1.0)
        err = np.abs(mapper.transform(theta) - stored) / sigma
        self.assertLess(err.max(), mapper.TOL)
        self.assertIn(mapper.method, {"analytic", "fallback"})

    def test_mapper_applies_to_unseen_parameter_sets_without_error(self):
        mapper = fp.GridMapper(self.data)
        lo = self.data["cases"][proxy.PARAMS].min().to_numpy()
        hi = self.data["cases"][proxy.PARAMS].max().to_numpy()
        rng = np.random.default_rng(0)
        theta = lo + rng.random((5, 4)) * (hi - lo)
        out = mapper.transform(theta)
        self.assertEqual(out.shape, (5, 8))
        self.assertTrue(np.isfinite(out).all())


class MisfitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fp.load_cumulative_data()

    def test_perfect_prediction_has_zero_misfit(self):
        y = self.data["y_hist"]
        total, per_phase = fp.hm_misfit_parts(np.repeat(y[None], 3, axis=0), y)
        np.testing.assert_allclose(total, 0.0, atol=0)
        np.testing.assert_allclose(per_phase, 0.0, atol=0)

    def test_constant_relative_offset_is_the_misfit(self):
        # pred = obs + 0.1 * obs_final makes every normalized residual exactly 0.1
        y = self.data["y_hist"]
        total, per_phase = fp.hm_misfit_parts(
            np.repeat((y + 0.1 * y[-1])[None], 2, axis=0), y)
        np.testing.assert_allclose(total, 0.1, rtol=1e-12)
        np.testing.assert_allclose(per_phase, 0.1, rtol=1e-12)

    def test_case_misfits_match_manual_formula_on_a_known_case(self):
        totals, per_phase = fp.case_misfits(self.data)
        self.assertEqual(totals.shape, (100,))
        self.assertTrue(np.isfinite(totals).all() and (totals >= 0).all())
        for i in (0, 42, 99):
            d = (self.data["Y"][i] - self.data["y_hist"]) / self.data["y_hist"][-1]
            np.testing.assert_allclose(totals[i], np.sqrt((d ** 2).mean()), rtol=1e-12)
            for j in range(3):
                np.testing.assert_allclose(per_phase[i, j],
                                           np.sqrt((d[:, j] ** 2).mean()), rtol=1e-12)


class FamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fp.load_cumulative_data()
        cls.t = np.asarray(cls.data["t_days"], dtype=float)

    def test_backtest_smoke_returns_a_winner_per_phase(self):
        table, winners = fp.backtest_families(self.data["Y"][:8], self.t,
                                              n_fit=25, steps=150)
        self.assertEqual(len(table), 9)
        self.assertEqual(set(winners), set(proxy.PHASES))
        for family in winners.values():
            self.assertIn(family, fp.FAMILY_ORDER)
        self.assertTrue(np.isfinite(table["median_relative_rmse"]).all())
        self.assertTrue((table["median_relative_rmse"] > 0).all())

    def test_forecast_extends_curves_continuously_and_monotonically(self):
        n, n_future = 2, 80
        t_future = np.linspace(self.t[-1] + 91, self.t[-1] + 365 * 20, n_future)
        shape = 8e6 * (1.0 - np.exp(-(self.t / 3500.0) ** 1.2))
        C_hist = np.stack([shape * s for s in (1.0, 1.4)])[:, :, None] * np.ones((1, 1, 3))
        winners = {phase: "gompertz" for phase in proxy.PHASES}
        fut = fp.forecast_from_candidates(C_hist, self.t, t_future, winners, steps=200)
        self.assertEqual(fut.shape, (n, n_future, 3))
        self.assertTrue(np.isfinite(fut).all())
        self.assertTrue((fut >= C_hist[:, -1:, :] - 1e-9).all())
        self.assertTrue((np.diff(fut, axis=1) >= -1e-9).all())


class SelectionTests(unittest.TestCase):
    def test_select_top10_picks_one_best_member_per_cluster(self):
        rng = np.random.default_rng(0)
        centers = np.array([[0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.5, 0.5], [0.9, 0.9, 0.9, 0.9]])
        theta = np.vstack([c + rng.normal(0, 0.005, (10, 4)) for c in centers])
        score = rng.random(len(theta))
        sel, labels = fp.select_top10(theta, score, k=10)
        self.assertEqual(len(sel), len(set(sel.tolist())))
        self.assertEqual(len(sel), 10)
        np.testing.assert_array_equal(score[sel], np.sort(score[sel])[::-1])
        for i in sel:
            same = labels == labels[i]
            self.assertEqual(score[i], score[same].max())


class DeliverableTests(unittest.TestCase):
    def test_writer_fills_10_sheets_and_passes_readback(self):
        n = 80
        selected = []
        for rank in range(1, 11):
            ramp = np.arange(n) * 1e5 + 5e6 + 1e4 * rank
            selected.append({"rank": rank, "gas": ramp * 0.7, "oil": ramp, "water": ramp * 0.4})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "deliverable.xlsx"
            report = fp.write_deliverable(selected, out)
            self.assertEqual(len(report["sheets"]), 10)
            for checks in report["sheets"].values():
                self.assertTrue(checks["ok"], checks)
            template = fp.TEMPLATE
            for rank in range(1, 11):
                name = f"Case {rank}"
                written = pd.read_excel(out, sheet_name=name, header=None)
                expected = pd.read_excel(template, sheet_name=name, header=None)
                self.assertEqual(written.shape[0], 81)   # header + 80 quarters
                pd.testing.assert_series_equal(written.iloc[1:, 0].reset_index(drop=True),
                                               expected.iloc[1:, 0].reset_index(drop=True),
                                               check_names=False)
                np.testing.assert_allclose(written.iloc[1:, 1].to_numpy(dtype=float),
                                           selected[rank - 1]["gas"], rtol=0, atol=0.5)
                np.testing.assert_allclose(written.iloc[1:, 2].to_numpy(dtype=float),
                                           selected[rank - 1]["oil"], rtol=0, atol=0.5)
                np.testing.assert_allclose(written.iloc[1:, 3].to_numpy(dtype=float),
                                           selected[rank - 1]["water"], rtol=0, atol=0.5)


class CumulativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fp.load_cumulative_data()

    def test_smoke_train_three_seeds_thirty_epochs(self):
        cfg = proxy.Config(feature_set="p4g8", hidden=(16, 16), dropout=0.0,
                           weight_decay=0.0, max_epochs=30, patience=30)
        states, scaler, epochs = proxy.train_final(self.data, cfg, n_seeds=3, epochs=30)
        self.assertEqual(epochs, 30)
        X = proxy.build_inputs(self.data, cfg.feature_set)
        pred = proxy.predict_ensemble(states, scaler, X, cfg)
        self.assertEqual(pred.shape, (100, 41, 3))
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= 0).all())

    @unittest.skipUnless(fp.CUM_MODEL.exists(), "cumulative_model.pt not trained yet")
    def test_checkpoint_predicts_stored_cases_better_than_the_mean_curve(self):
        ckpt = fp.load_cumulative_model()
        theta = self.data["cases"][proxy.PARAMS].to_numpy(dtype=np.float64)
        pred = fp.predict_cumulative(ckpt, theta)
        self.assertEqual(pred.shape, (100, 41, 3))
        self.assertTrue(np.isfinite(pred).all())
        test = self.data["cases"]["split"].to_numpy() == "test"
        rmse = np.sqrt(np.mean((pred[test] - self.data["Y"][test]) ** 2))
        mean_curve = np.repeat(
            self.data["Y"][self.data["cases"]["split"].to_numpy() == "train"].mean(axis=0)[None],
            test.sum(), axis=0)
        rmse_mean = np.sqrt(np.mean((mean_curve - self.data["Y"][test]) ** 2))
        self.assertLess(rmse, 0.5 * rmse_mean)


if __name__ == "__main__":
    unittest.main()
