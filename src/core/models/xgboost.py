"""Base trainer class for all model trainers."""

import logging
from abc import ABC, abstractmethod
import os
from typing import Any, List, Optional
from pathlib import Path
import pandas as pd
import joblib
from src.core.config_loader import ConfigLoader
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Non-interactive backend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XgboostModel():
    """Abstract base class for model trainers."""
    
    def __init__(self):
        """
        Initialize base trainer.
        
        Args:
            model_name: Model configuration name.
        """
        self.config = ConfigLoader()
        self.model_params = self.config.params.get("xgboost")
        
        # Create run directory (this sets self.run_directory)
        self._get_run_directory()
        
        # Get model algorithm
        self.model = xgb.XGBRegressor
    

    # Todo: Consider enablingmore than one validation sets here.
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[List[pd.DataFrame]] = None,
        y_val: Optional[List[pd.Series]] = None
    ) -> Any:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained model
        """
        # Set training and eval data attibutes
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        logger.info(
            """
            =================================================
            Training XGBoost Model...
            =================================================
            """
        )

        # Initialize model
        self.model = self.model(
            **self.model_params
        )

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            for x, y in zip(X_val, y_val):
                eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True
        )

        # Save model
        self._save_model()

        # Save feature importance
        self._get_feature_importance()


        # save configurations dataframe
        self.config.extract_config_dataframe().to_csv(
            os.path.join(self.run_directory, "config_params.csv"),
            index=False
        )

        logger.info(
            f"""
            =================================================
            Model Training Pipeline completed successfully!
            Run ID: {self.run_directory}
            Artifacts saved to: {self.run_directory}
            =================================================
            """
        )

    def predict(self, X_val: pd.DataFrame) -> pd.Series:
        """
        Make predictions using trained model.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        predictions = self.model.predict(X_val)
        return pd.Series(predictions, index=X_val.index)

    def _get_run_directory(self) -> str:
        """This function creates the directory for the model that was just trained"""
        self.models_directory_path = os.path.join(Path(__file__).parent.parent.parent, "artifacts", "models")
        self.run_directory = os.path.join(self.models_directory_path, self.model_name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_directory, exist_ok=True)
    

    def _save_model(self, model_index: int = 0):
        """
        Save trained model to disk.

        Args:
            model_path: Path to save model
        """
        model_path = os.path.join(self.run_directory, f"model_{model_index}.joblib")
        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


    def load_model(self, model_path: str):
        """
        Load model from disk.

        Args:
            model_path: Path to model file
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _get_feature_importance(self) -> None:
        """
        Get feature importance from XGBoost model.
        
        Returns:
            DataFrame with feature names and importance scores
        """

        importance_dict = self.model.get_booster().get_score(importance_type='weight')

        self.feature_importance = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)

        importance_path = os.path.join(self.run_directory, 'feature_importance.csv')
        self.feature_importance.to_csv(importance_path, index=False)
    

        
