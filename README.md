# Australian Rain Prediction & Model Explainability

A comprehensive Machine Learning pipeline in Python that predicts whether it will rain tomorrow in Australia. The project covers data preprocessing, advanced missing value imputation, öznitelik seçimi (feature selection), classifier training, and model explainability using permutation importance.

---

## 🚀 Key Features

* **MICE Imputation**: Implements **Multivariate Imputation by Chained Equations (MICE)** using `IterativeImputer` to reconstruct missing values in weather attributes (such as temperature, wind speed, pressure, and humidity).
* **Class Imbalance Resolution**: Utilizes resampling techniques to manage skewed classification categories.
* **Feature Selection**: Employs `SelectKBest` and Random Forest `SelectFromModel` to identify the most predictive feature subsets.
* **Model Explainability (ELI5)**: Computes **Permutation Importance** using the `eli5` library to calculate the feature weights contributing to predictions, providing clarity on model decisions.
* **Classifier Modeling**: Trains multiple binary classifiers (including Random Forest and XGBoost) to optimize predicting target labels.

---

## 📁 Repository Structure

```
australian-rain-prediction/
├── Aus_P.ipynb          ← Jupyter Notebook executing data loading, MICE, training & ELI5
├── weatherAUS.csv       ← Weather dataset containing daily observations across Australia
├── LICENSE              ← MIT License
├── requirements.txt     ← Project dependencies
├── .gitignore           ← Git ignore file
└── README.md            ← Project documentation (This file)
```

---

## 📦 Requirements & Dependencies

The project requires Python 3.8+ and the following libraries:

* **numpy** >= 1.22.0
* **pandas** >= 1.4.0
* **matplotlib** >= 3.5.0
* **seaborn** >= 0.11.0
* **scikit-learn** >= 1.0.0
* **eli5** >= 0.13.0
* **xgboost** >= 1.6.0

---

## ⚙️ Quick Start

### Prerequisites
* Python 3.8 or higher installed.

### Installation
1. Clone the repository to your local directory:
   ```bash
   git clone https://github.com/MustafArikan/australian-rain-prediction.git
   cd australian-rain-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
Open Jupyter Notebook to run the cells sequentially:
```bash
jupyter notebook Aus_P.ipynb
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
