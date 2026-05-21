import unittest

from src.cohort import build_new_user_active_comparator_cohort
from src.covariates import baseline_covariate_columns
from src.data_model import load_config
from src.propensity import balance_diagnostics, estimate_propensity_scores, max_abs_smd
from src.synthetic_data import generate_synthetic_data


class PropensityBalanceTests(unittest.TestCase):
    def test_weighting_improves_balance_and_keeps_ess(self):
        config = load_config("config/research_question.yaml")
        synthetic = generate_synthetic_data(config, n=2200, seed=41)
        cohort, _ = build_new_user_active_comparator_cohort(synthetic.tables, config)
        covars = baseline_covariate_columns(cohort, include_proxy=False)
        unweighted = balance_diagnostics(cohort, covars, weight_col=None)
        ps = estimate_propensity_scores(cohort, covars)
        weighted_max = max_abs_smd(ps.balance)
        unweighted_max = max_abs_smd(unweighted)
        self.assertLess(weighted_max, unweighted_max)
        self.assertLess(weighted_max, 0.12)
        self.assertGreater(ps.diagnostics["effective_sample_size_ratio"], 0.70)


if __name__ == "__main__":
    unittest.main()
