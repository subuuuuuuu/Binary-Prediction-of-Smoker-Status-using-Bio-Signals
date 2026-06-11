import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from src.utils import categorize_age, BMI, eye

# 1. RAW COLUMNS (Simple APPROACH)
RAW_FEATURE_COLS = [
    'age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
    'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic',
    'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
    'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine',
    'AST', 'ALT', 'Gtp', 'dental caries'
]

# 2. FEATURE ENGINEERED COLUMNS
NUM_WITH_OUTLIER = [
    'eyesight', 'hearing', 'fasting blood sugar', 'LDL', 'Urine protein',
    'AST', 'ALT', 'Gtp'
]

NUM_WITHOUT_OUTLIER = [
    'systolic', 'relaxation', 'Cholesterol', 'triglyceride', 'HDL',
    'hemoglobin', 'serum creatinine', 'dental caries'
]

CAT_COLS = ['age', 'BMI']


def load_data(filepath, is_train=True):
    """Loads CSV data, extracts features and target if applicable."""
    df = pd.read_csv(filepath)
    if is_train:
        X = df.drop(columns=['smoking', 'id'], errors='ignore')
        y = df['smoking'] if 'smoking' in df.columns else None
        return X, y
    else:
        X = df.drop(columns=['id'], errors='ignore')
        return X


def add_engineered_features(df):
    """Applies the feature engineering from the exploratory phase:
    - Age categorization
    - BMI categorization
    - Eyesight average
    - Hearing average
    """
    df_new = df.copy()
    
    # Categorize age
    df_new['age'] = df_new['age'].apply(categorize_age)
    
    # Compute BMI
    df_new['BMI'] = df_new[['height(cm)', 'weight(kg)']].apply(
        lambda x: BMI(x['height(cm)'], x['weight(kg)']), axis=1
    )
    
    # Eyesight average
    df_new['eyesight'] = df_new[['eyesight(left)', 'eyesight(right)']].apply(
        lambda x: eye(x['eyesight(left)'], x['eyesight(right)']), axis=1
    )
    
    # Hearing average
    df_new['hearing'] = df_new[['hearing(left)', 'hearing(right)']].apply(
        lambda x: eye(x['hearing(left)'], x['hearing(right)']), axis=1
    )
    
    return df_new


def get_simple_preprocessor():
    """Returns a simple ColumnTransformer that applies StandardScaler to all raw features."""
    preprocessor = ColumnTransformer([
        ("Num_val", StandardScaler(), RAW_FEATURE_COLS)
    ], remainder='drop')
    return preprocessor


def get_engineered_preprocessor():
    """Returns the more advanced ColumnTransformer containing log transformations,
    scaling, and one-hot encoding for the engineered feature set.
    """
    pipe_num_without_outlier = make_pipeline(
        StandardScaler()
    )
    
    pipe_num_outlier = make_pipeline(
        FunctionTransformer(np.log1p, validate=False), # np.log1p avoids log(0) issues if any
        StandardScaler()
    )
    
    pipe_cat_val = make_pipeline(
        OneHotEncoder(handle_unknown='ignore')
    )
    
    preprocessor = ColumnTransformer([
        ("Num_val", pipe_num_outlier, NUM_WITH_OUTLIER),
        ("Num_val_outlier", pipe_num_without_outlier, NUM_WITHOUT_OUTLIER),
        ("cat_val", pipe_cat_val, CAT_COLS)
    ], remainder='drop')
    
    return preprocessor
