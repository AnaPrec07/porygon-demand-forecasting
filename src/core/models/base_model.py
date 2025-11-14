"""Base trainer class for all model trainers."""

import logging
from abc import ABC, abstractmethod
import os
from typing import Any, Optional
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


class BaseTrainer():
    """Abstract base class for model trainers."""
    
    def __init__(self, model_name: str):
        """
        Initialize base trainer.
        
        Args:
            model_name: Model configuration name.
        """
        self.model_name = model_name
        self.config = ConfigLoader()
        self.model_params = self.config.params.get(self.model_name)
        
        # Create run directory (this sets self.run_directory)
        self._get_run_directory()
        
        # Get model algorithm
        self.model = self.get_model_algorithm()
    
    def get_model_algorithm(self):
        """Get model algorithm."""
        models_dictionary = {
            "xgboost": xgb.XGBRegressor,
        }
        return models_dictionary.get(self.model_name)


    #@abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
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

        logger.info(f"Training {self.model_name} model...")
        # todo: here I want to print all the training configurations in the config file.

        # Initialize model
        self.model = self.model(
            **self.model_params
        )

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=True
        )

        # Store training history
        self.training_history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
        }

        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        logger.info(f"Best score: {self.model.best_score:.4f}")

        # Save model
        self._save_model()

        # Save feature importance
        self._get_feature_importance()

        # Generate evaluation artifacts
        self._generate_eval_artifacts()

        # save configurations dataframe
        self.config.get_config_dataframe().to_csv(
            os.path.join(self.run_directory, "config_params.csv"),
            index=False
        )

        # Create visualizations.
        self.create_visualization()

        logger.info(
            f"""
            =================================================
            Model Training Pipeline completed successfully!
            Run ID: {self.run_directory}
            Artifacts saved to: {self.run_directory}
            =================================================
            """
        )


    def _generate_eval_artifacts(self):
        """Produce evaluation artifacts."""
        logger.info("Generating predictions on eval sets...")
        self.pred_train = self.predict(self.X_train)
        self.pred_val = self.predict(self.X_val)

        self.eval_metrics_df = pd.DataFrame({
            "set": ["train", "val"],
            "mae": [float(mean_absolute_error(self.y_train, self.pred_train)), float(mean_absolute_error(self.y_val, self.pred_val))],
            "rmse": [float(np.sqrt(mean_squared_error(self.y_train, self.pred_train))), float(np.sqrt(mean_squared_error(self.y_val, self.pred_val)))],
            "r2": [float(r2_score(self.y_train, self.pred_train)), float(r2_score(self.y_val, self.pred_val))],
            "mape": [float(np.mean(np.abs((self.y_train - self.pred_train) / self.y_train)) * 100), float(np.mean(np.abs((self.y_val - self.pred_val) / self.y_val)) * 100)],
        })

        self.eval_metrics_df.to_csv(
            os.path.join(self.run_directory, "evaluation_metrics.csv"),
            index=False
        )

        predictions_df = pd.DataFrame({
            'actual': self.y_val.values,
            'predicted': self.pred_val.values,
            'residual': self.y_val.values - self.pred_val.values,
            'absolute_error': np.abs(self.y_val.values - self.pred_val.values),
        })
        predictions_path = os.path.join(self.run_directory, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"✓ Predictions saved: {predictions_path}")
    
    def create_visualization(self):
        """Create evaluation visualizations."""
        # 7. Create visualizations (optional)
        try:
            
            # Actual vs Predicted plot
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Actual vs Predicted scatter
            axes[0, 0].scatter(self.y_val, self.pred_val, alpha=0.5)
            axes[0, 0].plot([self.y_val.min(), self.y_val.max()], [self.y_val.min(), self.y_val.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Residuals
            residuals = self.y_val.values - self.pred_val.values
            axes[0, 1].scatter(self.pred_val, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residual Plot')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Residuals distribution
            axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residuals Distribution')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Feature importance (top 15)
            if self.feature_importance is not None and len(self.feature_importance) > 0:
                top_features = self.feature_importance.head(15)
                axes[1, 1].barh(range(len(top_features)), top_features['importance'])
                axes[1, 1].set_yticks(range(len(top_features)))
                axes[1, 1].set_yticklabels(top_features['feature'])
                axes[1, 1].set_xlabel('Importance')
                axes[1, 1].set_title('Top 15 Feature Importance')
                axes[1, 1].invert_yaxis()
                axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            # Save figure (multiple formats for flexibility)
            plots_path_png = os.path.join(self.run_directory, "evaluation_plots.png")
            plots_path_pdf = os.path.join(self.run_directory, "evaluation_plots.pdf")
            
            # Save as PNG (for quick viewing)
            fig.savefig(plots_path_png, dpi=150, bbox_inches='tight', facecolor='white')
            logger.info(f"✓ Evaluation plots saved: {plots_path_png}")
            
            # Optionally save as PDF (for publications/presentations)
            fig.savefig(plots_path_pdf, bbox_inches='tight', facecolor='white')
            logger.info(f"✓ Evaluation plots saved: {plots_path_pdf}")
            
            # Close figure to free memory
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
            import traceback
            logger.debug(traceback.format_exc())



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
        
