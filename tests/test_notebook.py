import json
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CELLS = json.loads((ROOT / "Aus_P.ipynb").read_text(encoding="utf-8"))["cells"]


class NotebookRegressionTests(unittest.TestCase):
    def test_bundled_dataset_loads_from_repository_root(self):
        original = Path.cwd()
        try:
            os.chdir(ROOT)
            namespace = {}
            exec("".join(CELLS[0]["source"]), namespace)
            self.assertIn("RainTomorrow", namespace["full_df"].columns)
            self.assertGreater(len(namespace["full_df"]), 0)
        finally:
            os.chdir(original)

    def test_test_set_uses_training_scaler(self):
        columns = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation",
                   "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am",
                   "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
                   "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
                   "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"]
        frame = pd.DataFrame(np.random.default_rng(42).normal(size=(40, len(columns))), columns=columns)
        frame["RainTomorrow"] = np.arange(40) % 2
        namespace = {"MiceImputed": frame}
        exec("".join(CELLS[24]["source"]), namespace)
        train, test = train_test_split(frame[columns], test_size=0.25, random_state=12345)
        scaler = StandardScaler().fit(train)
        np.testing.assert_allclose(namespace["X_train"], scaler.transform(train))
        np.testing.assert_allclose(namespace["X_test"], scaler.transform(test))


if __name__ == "__main__":
    unittest.main()
