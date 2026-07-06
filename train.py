import argparse
import random
import torch
import os
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.models import model_attention
from src.preprocessing import Preprocessor

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="random seed for reproducibility")
parser.add_argument("--exp_name", default="probability_preds", type=str)
parser.add_argument("--save_dir", default="checkpoints", type=str)

parser.add_argument("--dataset_file", default="fve_aba_dataset.csv", type=str, help="name of .csv file with dataset")
parser.add_argument("--train_ratio", default=0.7, type=float, help="portion of data for training")
parser.add_argument("--val_ratio", default=0.15, type=float, help="portion of data for validation")

parser.add_argument("--target_col", default="energy", type=str, help="predicted variable")
parser.add_argument("--lookback_cols", nargs="+", 
    default=[
        "cloud_cover.total", "pressure", "irradiance", "ozone", "humidity", 
        "openmeteo_pm10", "tmp_module", "wind_u", "wind_v", "solar_elevation", 
        "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year", "energy",
    ], 
    help="List of features for history (lookback)"
)
parser.add_argument("--horizon_cols", nargs="+", 
    default=[
        "cloud_cover.total", "pressure", "irradiance", "ozone", "humidity", 
        "openmeteo_pm10", "temperature", "wind_u", "wind_v", "solar_elevation", 
        "sin_hour", "cos_hour", "sin_day_of_year", "cos_day_of_year",
    ], 
    help="List of features for future horizon"
)

parser.add_argument("--batch_size", default=128, type=int, help="size of batch")
parser.add_argument("--epochs", default=28, type=int, help="number of training epochs")
parser.add_argument("--learning_rate", default=0.0006347523354663052, type=float, help="learning rate")
parser.add_argument("--weight_decay", default=0.0008413987058716558, type=float, help="weight_decay")
parser.add_argument("--eta_min", default=1e-6, type=float, help="Minimum learning rate for Cosine Annealing scheduler")

# --- HistoryEncoder (Past) ---
parser.add_argument("--past_hidden_size", default=16, type=int, help="hidden states in HistoryEncoder LSTM")
parser.add_argument("--past_cnn_filters", default=16, type=int, help="number of filters in cnn in HistoryEncoder")
parser.add_argument("--past_dropout", default=0.4402635671482907, type=float, help="Dropout rate in HistoryEncoder")
parser.add_argument("--past_kernel", default=7, type=float, help="Dropout rate in HistoryEncoder")

# --- FutureEncoder (Future) ---
parser.add_argument("--future_hidden_size", default=32, type=int, help="hidden states in FutureEncoder LSTM")
parser.add_argument("--future_cnn_filters", default=32, type=int, help="number of filters in cnn in FutureEncoder")
parser.add_argument("--future_kernel_L0", default=5, type=int, help="first cnn's kernel")
parser.add_argument("--future_kernel_L1", default=3, type=int, help="second cnn's kernel")
parser.add_argument("--future_dropout", default=0.08649601155033557, type=float, help="Dropout rate in FutureEncoder")

# --- Decoder & Attention ---
parser.add_argument("--attention_dim", default=128, type=int, help="Dimensionality of the attention projection (internal attention space)")
parser.add_argument("--decoder_dropout", default=0.07088107433700343, type=float, help="Dropout rate inside the Decoder")

parser.add_argument("--lookback", default=24, type=int, help="number of past rows (hours) to look back for historical weather and energy data")
parser.add_argument("--horizon", default=24, type=int, help="number of future rows (hours) to look ahead for weather forecast and target predictions")

# ---Model --- TODO try give the model instead of median all probabs
parser.add_argument("--quantiles", default=[0.1, 0.3, 0.5, 0.7, 0.9], nargs='+', type=float, help="list of quantiles.")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_dataset(dataset_file: str, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(f"data/{dataset_file}", index_col="timestamp", parse_dates=True)
    dataset = dataset.sort_index().asfreq('1h')
    
    total_len = len(dataset)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_df = dataset.iloc[:train_end].copy()
    val_df = dataset.iloc[train_end:val_end].copy()
    test_df = dataset.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def prepare_and_scale_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Preprocessor]:
    """
    Combines features, normalizes all data splits, and saves scalers to disk.
    Returns the scaled dataframes and the preprocessor instance for later inverse transformation.
    """
    feature_cols = list(set(args.lookback_cols + args.horizon_cols))
    
    preprocessor = Preprocessor(feature_cols=feature_cols, target_col=args.target_col)
    train_scaled, val_scaled, test_scaled = preprocessor.process_data(train_df=train_df, val_df=val_df, test_df=test_df)
    
    preprocessor.save_scalers(save_dir=args.save_dir)
    print("Data successfully normalized and scalers saved.")
    
    return train_scaled, val_scaled, test_scaled, preprocessor

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{args.exp_name}_{timestamp}")

    train_df, val_df, test_df = load_dataset(args.dataset_file, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    train_df, val_df, test_df, preprocessor = prepare_and_scale_data(train_df, val_df, test_df, args)

    model = model_attention.Model(args=args).to(device)
    pass

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
