import unittest

import numpy as np

import proxy_model as proxy


class ProxyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = proxy.load_data()

    def test_data_contains_the_expected_training_validation_and_test_cases(self):
        cases = self.data["cases"]
        self.assertEqual(cases["split"].value_counts().to_dict(), {
            "train": 70,
            "validation": 15,
            "test": 15,
        })
        train_ids = cases.loc[cases["split"] == "train", "case_id"].to_numpy()
        np.testing.assert_array_equal(train_ids, np.arange(1, 71))
        self.assertEqual(self.data["Y"].shape, (100, 41, 3))
        self.assertEqual(len(self.data["dates"]), 41)
        self.assertTrue(np.isfinite(self.data["Y"]).all())
        self.assertTrue((self.data["Y"] >= 0).all())

    def test_four_uncertainty_and_optional_grid_feature_inputs_are_finite(self):
        p4 = proxy.build_inputs(self.data, "p4")
        p4g8 = proxy.build_inputs(self.data, "p4g8")
        self.assertEqual(p4.shape, (100, 4))
        self.assertEqual(p4g8.shape, (100, 12))
        self.assertTrue(np.isfinite(p4g8).all())
        train_mask = self.data["cases"]["split"].to_numpy() == "train"
        scaled_columns = [f"{column}_s" for column in proxy.PARAMS]
        np.testing.assert_allclose(
            p4[train_mask],
            self.data["cases"].loc[train_mask, scaled_columns].to_numpy(),
            atol=1e-12,
        )

    def test_phase_scaler_round_trips_rates(self):
        rates = self.data["Y"][:70]
        scaler = proxy.PhaseScaler(use_log=True).fit(rates)
        restored = scaler.inverse(scaler.transform(rates))
        np.testing.assert_allclose(restored, rates, rtol=2e-5, atol=2e-5)

    def test_network_smoke_training_predicts_all_dates_and_phases(self):
        cases = self.data["cases"]
        train_ids = np.flatnonzero(cases["split"].to_numpy() == "train")
        train_indices, validation_indices = train_ids[:-10], train_ids[-10:]
        config = proxy.Config(
            hidden=(8, 8),
            dropout=0.0,
            max_epochs=4,
            patience=2,
            weight_decay=0.0,
        )
        inputs = proxy.build_inputs(self.data, config.feature_set)
        scaler = proxy.PhaseScaler(config.use_log).fit(self.data["Y"][train_indices])
        targets = scaler.transform(self.data["Y"])
        state, epochs, loss = proxy._train_one(
            inputs[train_indices],
            targets[train_indices],
            inputs[validation_indices],
            targets[validation_indices],
            config,
            seed=proxy.SEED,
        )
        predicted = scaler.inverse(
            proxy.predict(state, inputs[validation_indices], inputs.shape[1], config)
        )
        self.assertGreaterEqual(epochs, 1)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(predicted.shape, (10, 41, 3))
        self.assertTrue(np.isfinite(predicted).all())


if __name__ == "__main__":
    unittest.main()
