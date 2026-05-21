import unittest

from src.cohort import build_new_user_active_comparator_cohort
from src.covariates import assert_no_post_index_covariates, baseline_covariate_columns
from src.data_model import load_config
from src.hdps import assert_hdps_pre_index, build_hdps_features
from src.synthetic_data import generate_synthetic_data


class NoPostIndexCovariateTests(unittest.TestCase):
    def test_baseline_covariates_exclude_post_index_columns(self):
        config = load_config("config/research_question.yaml")
        synthetic = generate_synthetic_data(config, n=1000, seed=31)
        cohort, _ = build_new_user_active_comparator_cohort(synthetic.tables, config)
        covars = baseline_covariate_columns(cohort, include_proxy=True)
        self.assertNotIn("post_index_healthcare_visits", covars)
        assert_no_post_index_covariates(cohort, covars)

    def test_hdps_restricts_to_pre_index_codes(self):
        config = load_config("config/research_question.yaml")
        synthetic = generate_synthetic_data(config, n=1000, seed=32)
        cohort, _ = build_new_user_active_comparator_cohort(synthetic.tables, config)
        _, ranking = build_hdps_features(cohort, synthetic.tables["code_history"], top_k=20)
        self.assertFalse(ranking["feature"].astype(str).str.contains("post_index").any())
        assert_hdps_pre_index(ranking, cohort)


if __name__ == "__main__":
    unittest.main()
