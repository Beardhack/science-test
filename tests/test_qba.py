import tempfile
import unittest
from pathlib import Path

from src.data_model import load_config
from src.quantitative_bias_analysis import binary_confounder_bias_factor, e_value, run_qba


class QuantitativeBiasAnalysisTests(unittest.TestCase):
    def test_e_value_and_binary_confounder_adjustment(self):
        self.assertAlmostEqual(e_value(1.0), 1.0)
        self.assertGreater(e_value(2.0), 3.0)
        bf = binary_confounder_bias_factor(0.20, 0.50, 2.0)
        self.assertLess(bf, 1.0)
        adjusted = 0.75 / bf
        self.assertGreater(adjusted, 0.75)

    def test_qba_outputs_table_and_plot(self):
        config = load_config("config/research_question.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            results, summary = run_qba(0.80, config, tmp)
            self.assertFalse(results.empty)
            self.assertIn("e_value", summary)
            self.assertTrue((Path(tmp) / "qba_results.csv").exists())
            self.assertTrue((Path(tmp) / "qba_tipping_point_plot.png").exists())


if __name__ == "__main__":
    unittest.main()
