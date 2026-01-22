import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.core.clients.bigquery import BigQueryClient
from src.core.config_loader import ConfigLoader
from core.models.xgboost import XgboostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for XGBoost training."""

    # Log Model Training
    logger.info(
        """
        =================================================
        XGBoost Model Training Pipeline is starting...
        =================================================
        """
    )
    
    # Initialize classes
    bq_client = BigQueryClient()
    config_loader = ConfigLoader()
    model_trainer = XgboostModel()

    # Set training fields to extract
    required_fields = ",".join(config_loader.required_fields)

    # Set features to use
    features = []

    # Extract training dataframe.
    loaded_df = bq_client.load_table(
        table_name=config_loader.training_table_name,
        fields=required_fields+features
    )

    # Filter Outliers
    logger.info(f"Filtering outliers ...")
    loaded_df = loaded_df[
        loaded_df[config_loader.target_col] < loaded_df[config_loader.target_col].quantile(config_loader.outlier_threshold)
    ]

    # Split train val.
    training_df = loaded_df[
        (loaded_df[config_loader.date_column] >= pd.to_datetime(config_loader.training_start_date))
        &  (loaded_df[config_loader.date_column] < pd.to_datetime(config_loader.train_end_date))
    ]
    validation_df = loaded_df[
        (loaded_df[config_loader.date_column] >= pd.to_datetime(config_loader.val_start_date))
        &  (loaded_df[config_loader.date_column] < pd.to_datetime(config_loader.val_end_date))
    ]

    X_train, y_train = training_df[config_loader.features_list], training_df[config_loader.target_col]
    X_val, y_val = validation_df[config_loader.features_list], validation_df[config_loader.target_col]

    
    # 01. Train model
    logger.info("Training model...")
    model_trainer.train(
        X_train,
        y_train,
        X_val=[X_val],
        y_val=[y_val]
    )
    


if __name__ == "__main__":
    main()
