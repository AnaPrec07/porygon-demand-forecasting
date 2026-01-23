import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.core.transformations.transformations import apply_log_normal_transformation, split_dataset
from src.core.clients.bigquery import BigQueryClient
from src.core.config_loader import ConfigLoader
from src.core.models.xgboost import XgboostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(features=["fea_dept_number"]):
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
    training_fields = ",".join(config_loader.required_fields + features)

    # Extract training dataframe.
    loaded_df = bq_client.load_training_table(
        table_name=config_loader.training_table_name,
        fields=training_fields
    )

    # Filter Outliers
    logger.info(f"Filtering outliers ...")
    loaded_df = loaded_df[
        loaded_df[config_loader.target_col] < loaded_df[config_loader.target_col].quantile(config_loader.outlier_threshold)
    ]
    
    # 01. Train model
    logger.info("Training model...")
    model_trainer.train(
        loaded_df,
        features
    )

    return model_trainer
    


if __name__ == "__main__":
    main()
