"""This module provides with reusable data transformation functions"""
from pandas import DataFrame, to_datetime
from src.core.config_loader import ConfigLoader
import numpy as np

config_loader = ConfigLoader()

def split_dataset(loaded_df: DataFrame):
    """This functions plits the loaded dataframe into training and validation sets based on 
    config file specifications."""

    # Split train val.
    training_df = loaded_df[
        (loaded_df[config_loader.date_column] >= to_datetime(config_loader.training_start_date))
        &  (loaded_df[config_loader.date_column] < to_datetime(config_loader.train_end_date))
    ]
    validation_df = loaded_df[
        (loaded_df[config_loader.date_column] >= to_datetime(config_loader.val_start_date))
        &  (loaded_df[config_loader.date_column] < to_datetime(config_loader.val_end_date))
    ]

    return training_df, validation_df

def apply_log_normal_transformation(loaded_df: DataFrame, target_col: str = config_loader.target_col):
    """This function applies log-normal transformation to the variable specified."""
    loaded_df[target_col] = np.log1p(loaded_df[target_col])
    return loaded_df

def reverse_log_normal_transformation(loaded_df: DataFrame, target_col: str = config_loader.target_col):
    """This function reverses the log-normal transformation to the variable specified."""
    loaded_df[target_col] = np.expm1(loaded_df[target_col])
    return loaded_df
