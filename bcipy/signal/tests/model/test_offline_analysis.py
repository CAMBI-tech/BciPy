"""Integration test of offline_analysis.py (slow)"""
from bcipy.signal.model import offline_analysis
import unittest
from bcipy.helpers.load import load_json_parameters
from pathlib import Path
import pytest
import shutil
import tempfile
import re
import numpy as np
import random
import gzip

pwd = Path(__file__).absolute().parent
input_folder = pwd / "integration_test_input"
expected_output_folder = pwd / "integration_test_expected_output"  # global for the purpose of pytest-mpl decorator


@pytest.mark.slow
class TestOfflineAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        random.seed(0)
        cls.tmp_dir = Path(tempfile.mkdtemp())

        # expand raw_data.csv.gz into tmp_dir
        with gzip.open(input_folder / "raw_data.csv.gz", "rb") as f_source:
            with open(cls.tmp_dir / "raw_data.csv", "wb") as f_dest:
                shutil.copyfileobj(f_source, f_dest)

        # copy the other required inputs into tmp_dir
        for f in ["triggers.txt", "parameters.json"]:
            shutil.copyfile(input_folder / f, cls.tmp_dir / f)

        cls.parameters = load_json_parameters(cls.tmp_dir / "parameters.json", value_cast=True)
        cls.model, fig_handles = offline_analysis(str(cls.tmp_dir), cls.parameters, alert_finished=False)
        cls.mean_erp_fig_handle, cls.lik_dens_fig_handle = fig_handles

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    @staticmethod
    def get_auc(model_filename):
        match = re.search("^model_([.0-9]+).pkl$", model_filename)
        if not match:
            raise ValueError()
        return float(match[1])

    def test_model_AUC(self):
        expected_auc = self.get_auc(list(expected_output_folder.glob("model_*.pkl"))[0].name)
        found_auc = self.get_auc(list(self.tmp_dir.glob("model_*.pkl"))[0].name)
        self.assertAlmostEqual(expected_auc, found_auc, delta=0.001)

    @pytest.mark.mpl_image_compare(baseline_dir=expected_output_folder, filename="test_mean_erp.png", remove_text=True)
    def test_mean_erp(self):
        return self.mean_erp_fig_handle

    @pytest.mark.mpl_image_compare(baseline_dir=expected_output_folder, filename="test_lik_dens.png", remove_text=True)
    def test_lik_dens(self):
        return self.lik_dens_fig_handle


if __name__ == "__main__":
    unittest.main()
