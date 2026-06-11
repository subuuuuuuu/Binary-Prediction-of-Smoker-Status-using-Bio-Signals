import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb

from src.preprocessing import (
    load_data,
    add_engineered_features,
    get_simple_preprocessor,
    get_engineered_preprocessor
)

def train_model(train_path, model_save_path, feature_engineering=False):
    print(f"Loading training data from {train_path}...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}. "
                                f"Please download the Kaggle smoker status dataset and place train.csv in data/raw/.")

    # 1. Load data
    X, y = load_data(train_path, is_train=True)
    
    # 2. Add features if feature engineering flag is set
    if feature_engineering:
        print("Applying feature engineering...")
        X = add_engineered_features(X)
        preprocessor = get_engineered_preprocessor()
    else:
        print("Using raw columns...")
        preprocessor = get_simple_preprocessor()

    # 3. Train-test split (matching the 80/20 split and random_state=0 from the notebook)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    print(f"Dataset split: {X_train.shape[0]} train samples, {X_val.shape[0]} validation samples.")

    # 4. Define pipeline matching the hyperparameters from cell 120 of the notebook
    print("Setting up model pipeline (LightGBM Classifier)...")
    clf = lgb.LGBMClassifier(
        num_leaves=128,
        min_child_samples=10,
        min_sum_hessian_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        random_state=0
    )
    
    pipeline = Pipeline([
        ("Pipeline", preprocessor),
        ("model", clf)
    ])

    # 5. Fit the model
    print("Training the pipeline...")
    pipeline.fit(X_train, y_train)

    # 6. Evaluate the model
    print("\n--- Evaluation on Validation Set ---")
    val_preds = pipeline.predict(X_val)
    val_probs = pipeline.predict_proba(X_val)[:, 1]

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, val_preds))
    print("\nClassification Report:")
    print(classification_report(y_val, val_preds))
    auc_score = roc_auc_score(y_val, val_probs)
    print(f"ROC-AUC Score: {auc_score:.5f}")

    # 7. Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    print(f"\nSaving the trained pipeline to {model_save_path}...")
    joblib.dump(pipeline, model_save_path)
    print("Training workflow completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Smoker Status Prediction model.")
    parser.add_argument(
        "--train-path", 
        type=str, 
        default=os.path.join("data", "raw", "train.csv"),
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=os.path.join("models", "model.joblib"),
        help="Path where the trained model.joblib will be saved"
    )
    parser.add_argument(
        "--feature-engineer", 
        action="store_true",
        help="Flag to enable feature engineering (BMI, eyesight avg, hearing avg, etc.)"
    )
    
    args = parser.parse_args()
    
    train_model(
        train_path=args.train_path,
        model_save_path=args.model_path,
        feature_engineering=args.feature_engineer
    )
