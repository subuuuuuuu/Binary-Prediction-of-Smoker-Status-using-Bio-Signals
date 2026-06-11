import os
import argparse
import joblib
import pandas as pd

from src.preprocessing import (
    load_data,
    add_engineered_features
)

def run_prediction(model_path, test_path, output_path, feature_engineering=False):
    # 1. Load the model
    print(f"Loading trained model pipeline from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run training first via 'python src/train.py'.")
    
    pipeline = joblib.load(model_path)
    print("Model pipeline loaded successfully.")

    # 2. Load the test dataset
    print(f"Loading test data from {test_path}...")
    if not os.path.exists(test_path):
        print("\n" + "="*80)
        print(f"WARNING: Test file not found at: {test_path}")
        print("To make predictions, please place your Kaggle test.csv file in the data/raw/ directory.")
        print("="*80 + "\n")
        return

    # Keep a copy for original IDs
    test_raw = pd.read_csv(test_path)
    ids = test_raw['id']

    X_test = load_data(test_path, is_train=False)

    # 3. Apply feature engineering if model expects it
    if feature_engineering:
        print("Applying feature engineering to test data...")
        X_test = add_engineered_features(X_test)

    # 4. Predict probabilities (smoker status target 'smoking' probability is y_hat[:, 1])
    print("Running inference...")
    try:
        # Check model steps to confirm alignment
        print("Pre-checks: checking feature columns compatibility...")
        probs = pipeline.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure the --feature-engineer flag matches how the model was trained.")
        return

    # 5. Save predictions in standard Kaggle submission format
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df = pd.DataFrame({
        'id': ids,
        'smoking': probs
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved successfully in correct format to: {output_path}")
    print(f"Total rows predicted: {len(submission_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on the Smoker Status test dataset.")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=os.path.join("models", "model.joblib"),
        help="Path to the trained model.joblib file"
    )
    parser.add_argument(
        "--test-path", 
        type=str, 
        default=os.path.join("data", "raw", "test.csv"),
        help="Path to the test.csv file"
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=os.path.join("data", "submissions", "smoker_status_submission.csv"),
        help="Path where output submission predictions will be saved"
    )
    parser.add_argument(
        "--feature-engineer", 
        action="store_true",
        help="Flag to enable feature engineering (must match the trained model pipeline)"
    )
    
    args = parser.parse_args()
    
    run_prediction(
        model_path=args.model_path,
        test_path=args.test_path,
        output_path=args.output_path,
        feature_engineering=args.feature_engineer
    )
