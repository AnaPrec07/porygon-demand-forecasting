import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.core.clients.bigquery import BigQueryClient
from src.core.config_loader import ConfigLoader
from core.models.xgboost import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for XGBoost training."""

    # Log Model Training
    logger.info(
        """
        =================================================
        Model Training Pipeline is starting...
        =================================================
        """
    )
    
    # Load data
    logger.info(f"Loading training data...")
    bq_client = BigQueryClient()
    config_loader = ConfigLoader()
    model_trainer = BaseTrainer(model_name="xgboost")

    # Set training fields to extract.
    fields = ",".join(config_loader.training_fields)

    # Extract training dataframe.
    loaded_df = bq_client.load_table(
        table_name=config_loader.training_table_name,
        fields=fields
    )
    loaded_df = loaded_df[~loaded_df[config_loader.target_col].isna()]

    if FILTER_POTENTIAL_STOCKOUT:
        logger.info(f"Filtering potential stockout rows with more than {STOCKOUT_DAYS_THRESHOLD} days of zero sales...")
        loaded_df = loaded_df[loaded_df[config_loader.potential_stockout_col] < STOCKOUT_DAYS_THRESHOLD]

    if FILTER_OUTLIERS:
        logger.info(f"Filtering outliers ...")
        loaded_df = loaded_df[loaded_df[config_loader.target_col] < 427]


    # Split train val.
    training_df = loaded_df[loaded_df[config_loader.date_column] < pd.to_datetime(config_loader.train_val_split_date)]
    validation_df = loaded_df[loaded_df[config_loader.date_column] >= pd.to_datetime(config_loader.train_val_split_date)]

    X_train, y_train = training_df[config_loader.features_list], training_df[config_loader.target_col]
    X_val, y_val = validation_df[config_loader.features_list], validation_df[config_loader.target_col]

    
    # 01. Train model
    logger.info("Training model...")
    model_trainer.train(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val
    )
    


if __name__ == "__main__":
    main()
