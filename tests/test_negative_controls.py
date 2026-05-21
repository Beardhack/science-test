import unittest

from src.cohort import build_new_user_active_comparator_cohort
from src.covariates import baseline_covariate_columns
from src.data_model import load_config
from src.negative_controls import negative_control_candidates, run_negative_control_outcome_analysis
from src.propensity import estimate_propensity_scores
from src.synthetic_data import generate_synthetic_data


class NegativeControlTests(unittest.TestCase):
    def test_candidates_are_marked_for_clinical_review(self):
        config = load_config("config/research_question.yaml")
        candidates = negative_control_candidates(config)
        self.assertFalse(candidates.empty)
        self.assertTrue(candidates["status"].str.contains("requires clinical review").all())

    def test_negative_control_analysis_runs(self):
        config = load_config("config/research_question.yaml")
        synthetic = generate_synthetic_data(config, n=1200, seed=51)
        cohort, _ = build_new_user_active_comparator_cohort(synthetic.tables, config)
        covars = baseline_covariate_columns(cohort, include_proxy=False)
        ps = estimate_propensity_scores(cohort, covars)
        results = run_negative_control_outcome_analysis(ps.data)
        self.assertIn("negative_control_adjusted", set(results["analysis"]))
        self.assertIn("residual_association_detected", results.columns)


if __name__ == "__main__":
    unittest.main()
