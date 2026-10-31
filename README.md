# Australian Rain Prediction

Explore Australian weather data, missing-value imputation, binary classification, and feature importance in a Jupyter notebook.

This research project uses the bundled `weatherAUS.csv` to investigate the `RainTomorrow` label. It is an exploratory experiment; its current preprocessing does not establish an independent estimate of forecasting performance.

## Key Features

- Weather-data exploration and class-distribution plots.
- Minority-class oversampling and iterative numeric imputation.
- Chi-squared and Random Forest feature selection.
- XGBoost classification, confusion matrices, ROC curves, and probability-based ROC AUC.
- ELI5 permutation importance and decision-region visualization with mlxtend.

## Tech Stack

Python, Jupyter, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, ELI5, and mlxtend. Runtime dependency ranges are in [requirements.txt](requirements.txt).

## Architecture & Project Structure

The notebook executes sequentially: load data, explore and resample, impute and filter, select features, split and scale, train, and evaluate.

```text
.
├── Aus_P.ipynb         # Analysis, modeling, and visualizations
├── weatherAUS.csv      # Bundled observations
├── requirements.txt   # Runtime dependencies
├── tests/             # Dataset-loading and scaling regression tests
└── LICENSE
```

**Methodological limitations:** oversampling, imputation, outlier filtering, and feature selection happen before the train/test split. Oversampling can place copies of an observation in both sets. Permutation importance is also computed on fitted data. These steps can inflate evaluation results; move learned preprocessing and resampling inside training folds before interpreting scores as generalization performance.

## Getting Started

Use Python 3.11 in an isolated environment. The dependency ranges are not a lockfile.

```bash
git clone https://github.com/MustafArikan/australian-rain-prediction.git
cd australian-rain-prediction
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
```

## Configuration

No environment variables or credentials are required. Start Jupyter from the repository root so `weatherAUS.csv` resolves correctly. The notebook contains modeling parameters, including the split seed and XGBoost settings.

## Usage

```bash
jupyter notebook Aus_P.ipynb
```

Run cells in order in a fresh kernel. Imputation, resampling, pair plots, and training can consume substantial memory and time. Results remain in notebook outputs unless explicitly exported.

## Testing

```bash
python -m unittest discover -s tests
```

The tests execute the notebook's loading and split/scaling cells. They verify that the bundled data loads and that test features use the scaler fitted on training data. They do not certify the full experiment's statistical validity.

## Development & Quality

Keep changes to notebook source focused and clear affected outputs when their results become stale. No application build or deployment pipeline is configured. Use `python -m pip check` to inspect installed dependency consistency.

## Roadmap

- Split original observations before any learned preprocessing or resampling.
- Evaluate on an untouched, preferably time-aware holdout and report reproducible metrics.
- Fit feature selection and imputation within cross-validation folds.
- Record dataset provenance and a tested dependency lock.

## Contributing

Open an issue or pull request with the experiment, random seeds, dependency versions, and validation performed. Preserve the target definition and distinguish historical outputs from newly reproduced results.

## License

Code is distributed under the [MIT License](LICENSE). Check the source dataset's terms before redistributing data.

## Author

[Mustafa Arıkan](https://github.com/MustafArikan)
