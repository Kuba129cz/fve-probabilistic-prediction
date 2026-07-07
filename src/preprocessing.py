# src/preprocessing.py
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """Handles feature and target scaling. 
    Features (including past target values) are scaled using feature_scaler.
    Target scaler is used strictly for inverse transforming predictions.
    """
    
    def __init__(self, feature_cols: list[str], target_col: str):
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        self.cols_to_scale = [
            col for col in self.feature_cols if not (col.startswith("sin_") or col.startswith("cos_"))
        ]
        
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits both scalers on the training dataset."""
        if self.cols_to_scale:
            self.feature_scaler.fit(train_df[self.cols_to_scale])

        self.target_scaler.fit(train_df[[self.target_col]])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataset using feature_scaler. 
        Since target_col is part of cols_to_scale, it gets scaled automatically.
        """
        df_scaled = df.copy()

        if self.cols_to_scale:
            df_scaled[self.cols_to_scale] = self.feature_scaler.transform(df[self.cols_to_scale])
        
        return df_scaled

    def process_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame = None) -> tuple:
        """Convenience method to fit on train and transform all provided splits."""
        self.fit(train_df)
        
        train_scaled = self.transform(train_df)
        val_scaled = self.transform(val_df)
        
        if test_df is not None:
            test_scaled = self.transform(test_df)
            return train_scaled, val_scaled, test_scaled
            
        return train_scaled, val_scaled

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """Converts model's scaled quantile predictions back to original units (e.g. kW)."""
        original_shape = scaled_targets.shape
        flattened = scaled_targets.reshape(-1, 1)
        rescaled = self.target_scaler.inverse_transform(flattened)
        return rescaled.reshape(original_shape)

    def save_scalers(self, save_dir: str = "checkpoints/scalers"):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.feature_scaler, f"{save_dir}/feature_scaler.pkl")
        joblib.dump(self.target_scaler, f"{save_dir}/target_scaler.pkl")
        joblib.dump(self.cols_to_scale, f"{save_dir}/scaled_features_list.pkl")

    def load_scalers(self, load_dir: str = "checkpoints/scalers"):
        self.feature_scaler = joblib.load(f"{load_dir}/feature_scaler.pkl")
        self.target_scaler = joblib.load(f"{load_dir}/target_scaler.pkl")
        self.cols_to_scale = joblib.load(f"{load_dir}/scaled_features_list.pkl")