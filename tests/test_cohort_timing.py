import unittest

from src.cohort import assert_valid_cohort_timing, build_new_user_active_comparator_cohort
from src.data_model import load_config
from src.synthetic_data import generate_synthetic_data


class CohortTimingTests(unittest.TestCase):
    def test_new_user_active_comparator_timing(self):
        config = load_config("config/research_question.yaml")
        synthetic = generate_synthetic_data(config, n=1000, seed=21)
        cohort, attrition = build_new_user_active_comparator_cohort(synthetic.tables, config)
        assert_valid_cohort_timing(cohort)
        self.assertFalse(cohort["person_id"].duplicated().any())
        self.assertTrue(cohort["A"].isin([0, 1]).all())
        self.assertTrue((cohort["covariate_window_end"] < cohort["time_zero"]).all())
        self.assertGreater(len(attrition), 5)
        self.assertEqual(int(attrition.iloc[-1]["remaining"]), len(cohort))


if __name__ == "__main__":
    unittest.main()
