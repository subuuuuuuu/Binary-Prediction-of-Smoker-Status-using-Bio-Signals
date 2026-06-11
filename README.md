# Binary Prediction of Smoker Status using Bio-Signals

This project provides a robust, structured, and modular solution for the Kaggle Playground Series competition: **Binary Prediction of Smoker Status using Bio-Signals**. The goal of this machine learning pipeline is to predict whether a person is a smoker or not based on various bio-signals (clinical features like age, waist circumference, blood parameters, etc.).

---

## 📁 Repository Structure

The project has been restructured into a standard Data Science directory layout:

```text
Binary-Prediction-of-Smoker-Status-using-Bio-Signals-main/
│
├── .gitignore                      # Excludes virtual environments, caches, data, and models from Git
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
│
├── data/                           # Project Datasets
│   ├── raw/                        # Original raw datasets (train.csv / test.csv)
│   └── submissions/                # Output prediction files for Kaggle submission
│       └── smoker_status_submission.csv # Renamed from 'Multi-Class Prediction of Obesity Risk.csv'
│
├── models/                         # Trained model binaries
│   └── model.joblib                # Serialized trained LightGBM pipeline
│
├── notebooks/                      # Jupyter Notebooks for exploration and analysis
│   └── exploratory_analysis.ipynb  # Re-named notebook with updated local relative paths
│
└── src/                            # Modularized Python source packages
    ├── __init__.py                 # Package initializer
    ├── utils.py                    # Helper functions (BMI, average eyesight/hearing, age categories)
    ├── preprocessing.py            # Preprocessing pipelines (standard scalers, column transformers)
    ├── train.py                    # Script to train, validate, and save the model pipeline
    └── predict.py                  # Script to run batch inference on test data and export submissions
```

---

## 🛠️ Setup and Installation

A local Python virtual environment is recommended to run the code in isolation.

### 1. Create a Virtual Environment
From the project root directory, run:
```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
* **On Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **On Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **On macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Required Dependencies
With the environment active, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Machine Learning Pipeline

The project includes modular python scripts inside `src/` to execute data ingestion, training, evaluation, and batch inference.

### 1. Train the Model
To train the LightGBM classifier on the raw features (which is the default approach matching the saved `model.joblib` pipeline):
```bash
python src/train.py
```

If you wish to enable feature engineering (incorporating Custom BMI categories, average eyesight/hearing features, and binned age categories):
```bash
python src/train.py --feature-engineer
```

The script will automatically:
1. Load `data/raw/train.csv`.
2. Split it into an 80/20 train-validation split (reproducible seed `random_state=0`).
3. Standard-scale features (and log-transform engineered features if the flag is active).
4. Fit the LightGBM model with hyper-parameters matched from the notebook.
5. Print performance metrics (Confusion Matrix, Precision/Recall/F1-Score, and ROC-AUC score).
6. Save the trained pipeline to `models/model.joblib`.

### 2. Generate Predictions (Batch Inference)
Once a model is trained and saved in `models/model.joblib`, you can generate predictions on a test set:
```bash
python src/predict.py
```

If your model was trained with feature engineering enabled, make sure to add the corresponding flag:
```bash
python src/predict.py --feature-engineer
```

* **Note:** If `data/raw/test.csv` is missing, the script will output a friendly warning prompting you to place the test set in `data/raw/`.

Predictions are automatically output to `data/submissions/smoker_status_submission.csv` in the correct Kaggle submission format (containing columns `id` and `smoking`).

---

## 📓 Jupyter Notebook Usage
The exploratory notebook is relocated to `notebooks/exploratory_analysis.ipynb`. The cell code is fully updated to load raw data relative to the `notebooks/` directory and write models and submission files cleanly into their respective directories:
- Reads train data from: `../data/raw/train.csv`
- Reads test data from: `../data/raw/test.csv`
- Saves model to: `../models/model.joblib`
- Exports submission predictions to: `../data/submissions/smoker_status_submission.csv`
