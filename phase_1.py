from prefect import flow, task, get_run_logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.stats import entropy as scipy_entropy

# LOAD DATA
@task
def load_data(path):
    logger = get_run_logger()
    df = pd.read_csv(path)
    
    logger.info(f"Loaded shape: {df.shape}")
    return df

# DATA QUALITY CHECK
@task
def check_data(df):
    logger = get_run_logger()
    
    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    
    logger.info(f"Missing values: {missing}")
    logger.info(f"Duplicates: {duplicates}")
    
    return df

# CLEANING
@task
def clean_data(df):
    logger = get_run_logger()
    
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicates removed: {before - len(df)}")
    
    before = len(df)
    df = df[df["amount"] > 0].copy()
    logger.info(f"Zero amount removed: {before - len(df)}")
    
    df = df.drop(columns=["isFlaggedFraud"], errors="ignore")
    df = df.drop(columns=["nameOrig"], errors="ignore")
    
    logger.info("Dropped unnecessary columns")
    
    return df

# FEATURE ENGINEERING
@task
def feature_engineering(df):
    logger = get_run_logger()
    
    df["balanceDiffOrig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    
    df["isDestMerchant"] = (df["nameDest"].str.startswith("M")).astype(int)
    df = df.drop(columns=["nameDest"], errors="ignore")
    
    df["origDrainedToZero"] = (
        (df["oldbalanceOrg"] > 0) & 
        (df["newbalanceOrig"] == 0)
    ).astype(int)
    
    logger.info("Feature engineering complete")
    
    return df

# TRANSFORM (ENCODING + SCALING)
@task
def transform_data(df):
    logger = get_run_logger()
    
    # One-hot encoding
    df = pd.concat([df.drop(columns=["type"]), 
                    pd.get_dummies(df["type"], prefix="type")], axis=1)
    
    # Scaling
    scale_cols = [
        "amount", "oldbalanceOrg", "oldbalanceDest",
        "balanceDiffOrig", "balanceDiffDest"
    ]
    
    scaler = RobustScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    logger.info("Encoding + scaling done")
    
    return df

# FEATURE SELECTION
@task
def feature_selection(df):
    logger = get_run_logger()
    
    cols_to_drop = ["newbalanceOrig", "newbalanceDest", "step_day"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    logger.info("Dropped multicollinear features")
    
    return df

# SAVE
@task
def save_data(df, path):
    df.to_csv(path, index=False)
    
    logger = get_run_logger()
    logger.info(f"Saved to {path}")

# FLOW
@flow(name="Phase 1")
def preprocessing_pipeline():
    
    raw_path = "datasets/paysim-dataset.csv"
    output_path = "datasets/phase_1/paysim-dataset.csv"
    
    df = load_data(raw_path)
    df = check_data(df)
    df = clean_data(df)
    df = feature_engineering(df)
    df = transform_data(df)
    df = feature_selection(df)
    
    save_data(df, output_path)

# RUN
if __name__ == "__main__":
    preprocessing_pipeline()