import unittest

import numpy as np

from src.data_model import load_config
from src.synthetic_data import generate_synthetic_data


class SyntheticDataTests(unittest.TestCase):
    def test_generator_contains_latent_u_proxies_and_truth(self):
        config = load_config("config/research_question.yaml")
        synthetic = generate_synthetic_data(config, n=900, seed=11)
        persons = synthetic.tables["person"]
        self.assertIn("latent_u", persons.columns)
        self.assertIn("frailty_proxy", persons.columns)
        self.assertIn("negative_control_event_365", persons.columns)
        self.assertIn("code_history", synthetic.tables)
        self.assertLess(synthetic.truth["true_365_risk_ratio"], 1.0)
        corr = np.corrcoef(persons["latent_u"], persons["a1c_proxy"])[0, 1]
        self.assertGreater(abs(corr), 0.15)

    def test_latent_confounding_strength_is_configurable(self):
        config = load_config("config/research_question.yaml")
        weak = generate_synthetic_data(config, n=1200, seed=12, u_treatment_strength=0.0, u_outcome_strength=0.0)
        strong = generate_synthetic_data(config, n=1200, seed=12, u_treatment_strength=1.2, u_outcome_strength=1.2)
        weak_gap = abs(weak.tables["person"].groupby("A")["latent_u"].mean().diff().iloc[-1])
        strong_gap = abs(strong.tables["person"].groupby("A")["latent_u"].mean().diff().iloc[-1])
        self.assertGreater(strong_gap, weak_gap)


if __name__ == "__main__":
    unittest.main()
