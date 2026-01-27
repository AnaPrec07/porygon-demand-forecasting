"""Base trainer class for all model trainers."""

import logging
from abc import ABC, abstractmethod
import os
from typing import Any, List, Optional
from pathlib import Path
import pandas as pd
import joblib
from src.core.transformations.transformations import apply_log_normal_transformation, split_dataset
from src.core.config_loader import ConfigLoader
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_pinball_loss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.core.config_loader import ConfigLoader
import seaborn as sns
import pickle


matplotlib.use('Agg')  # Non-interactive backend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_loader = ConfigLoader()


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
        self.model_name = "xgboost"
        self.plot_residuals_paths = None
        
        # Create run directory (this sets self.run_directory)
        self._get_run_directory()
        
        # Get model algorithm
        self.model = xgb.XGBRegressor
    

    # Todo: Consider enablingmore than one validation sets here.
    def train(
        self,
        loaded_df: pd.DataFrame,
        features: List[str]
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

        # Apply log normal transformation to target
        loaded_df = apply_log_normal_transformation(loaded_df=loaded_df)
        loaded_df = apply_log_normal_transformation(loaded_df, target_col =config_loader.benchmark_col)
        self.features = features

        # Split train val.
        self.training_df, self.validation_df = split_dataset(loaded_df= loaded_df)
        self.X_train, self.y_train = self.training_df[features], self.training_df[config_loader.target_col]
        self.X_val, self.y_val = self.validation_df[features], self.validation_df[config_loader.target_col]


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
        eval_set = [(self.X_train, self.y_train), (self.X_val, self.y_val)]

        # Train model
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=eval_set,
            verbose=True
        )


        # save configurations dataframe
        self.config.extract_config_dataframe().to_csv(
            os.path.join(self.run_directory, "config_params.csv"),
            index=False
        )

        # Plot bias variance tradeoff:
        self._plot_bias_variance_tradeoff()

        # Plot residuals:
        self._evaluate_residuals()
    
        # Save feature importance
        self._get_feature_importance()

        # Save model
        self._save_model()

        logger.info(
            f"""
            =================================================
            Model Training Pipeline completed successfully!
            Run ID: {self.run_directory}
            Artifacts saved to: {self.run_directory}
            =================================================
            """
        )
    
    def _plot_bias_variance_tradeoff(self):
        evals_result = self.model.evals_result()
        learning_curve0 = evals_result['validation_0']['quantile']
        learning_curve1 = evals_result['validation_1']['quantile']

        benchmark_training = mean_pinball_loss(
            self.training_df[config_loader.target_col], 
            self.training_df[config_loader.benchmark_col]
        )

        benchmark_validation = mean_pinball_loss(
            self.validation_df[config_loader.target_col], 
            self.validation_df[config_loader.benchmark_col]
        )

        plt.figure(figsize=(10, 5))
        plt.axhline(y=benchmark_validation, color="green", marker="o", label="validation benchmark")
        plt.axhline(y=benchmark_training, color="gray", marker="o", label="training benchmark")
        sns.lineplot(x=range(len(learning_curve0)), y=learning_curve0, label="train")
        sns.lineplot(x=range(len(learning_curve1)), y=learning_curve1, label="val")
        plt.xlabel('Iteration')
        plt.ylabel('Mean Pinball Loss Error')
        plt.title('Validation Quantile Error over Iterations')

        plot_path = os.path.join(self.run_directory, "learning_curve.png")
        plt.savefig(plot_path)
        self.plot_bias_variance_tradeoff_path = plot_path

    def _plot_residuals(self, actuals, predictions, benchmark, pred_residuals, bench_residuals, suffix):
        _, axs = plt.subplots(2,2, figsize=(12,12))

        # Actuals vs Predictions
        axs[0,0].scatter(actuals, predictions, alpha = 0.5)
        axs[0,0].set_xlabel("Actuals")
        axs[0,0].set_ylabel("Predictions")
        axs[0,0].set_title("Actuals vs Predictions")
        axs[0,0].grid(True)

        # Actuals vs Benchmark
        axs[0,1].scatter(actuals, benchmark, alpha = 0.5)
        axs[0,1].set_xlabel("Actuals")
        axs[0,1].set_ylabel("Benchmark")
        axs[0,1].set_title("Actuals vs Benchmark")
        axs[0,1].grid(True)

        # Residuals vs Predictions
        axs[1,0].scatter(pred_residuals, predictions, alpha = 0.5)
        axs[1,0].set_xlabel("Residuals")
        axs[1,0].set_ylabel("Predictions")
        axs[1,0].set_title("Residuals vs Predictions")
        axs[1,0].grid(True)

        # Residuals vs Benchamrk
        axs[1,1].scatter(bench_residuals, benchmark, alpha = 0.5)
        axs[1,1].set_xlabel("Residuals")
        axs[1,1].set_ylabel("Benchmark")
        axs[1,1].set_title("Residuals vs Benchmark")
        axs[1,1].grid(True)

        plt.tight_layout()

        plot_path = os.path.join(self.run_directory, f"residuals_{suffix}.png")
        plt.savefig(plot_path)
        self.plot_residuals_paths = plot_path



    def _evaluate_residuals(self):

        self.val_predictions = self.predict(self.validation_df[self.features], exp_transformation=False)
        self.val_residuals = self.validation_df[config_loader.target_col] - self.val_predictions
        self.val_bench_residuals = self.validation_df[config_loader.target_col] - self.validation_df[config_loader.benchmark_col]

        # self.train_predictions = self.predict(self.training_df[self.features])
        # self.train_residuals = self.training_df[config_loader.target_col] - self.training_df
        # self.train_bench_residuals = self.training_df[config_loader.target_col] - self.training_df[config_loader.benchmark_col]


        self._plot_residuals(
            actuals=self.validation_df[config_loader.target_col],
            predictions=self.val_predictions,
            benchmark=self.validation_df[config_loader.benchmark_col],
            pred_residuals=self.val_residuals,
            bench_residuals = self.val_bench_residuals,
            suffix="validation"
        )
        
        #self._plot_residuals(
        #    actuals=self.training_df[config_loader.target_col],
        #    predictions=self.train_predictions,
        #    benchmark=self.training_df[config_loader.benchmark_col],
        #    pred_residuals=self.train_residuals,
        #    bench_residuals = self.train_bench_residuals,
        #    suffix="training"
        #)



    def predict(self, X_val: pd.DataFrame, exp_transformation = True) -> pd.Series:
        """
        Make predictions using trained model.
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        predictions = self.model.predict(X_val)
        if exp_transformation:
            predictions = np.expm1(predictions)
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
        model_path = os.path.join(self.run_directory, f"model_{model_index}.pickle")
        try:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
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
            with open(model_path, "rb") as f:
                logger.info(f"Model loaded from {model_path}")
                return pickle.load(f)
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

        # Plot importances
        self._plot_importance()
    
    def _plot_importance(self) -> None:

        top_importance = self.feature_importance[:4]
        fig, axs = plt.subplots(len((top_importance["feature"].to_list())),2, figsize=(12,12))

        for ax_index, feature in zip(range(len((top_importance["feature"].to_list()))), top_importance["feature"].to_list()):
            # Feature vs Residual
            axs[ax_index, 0].scatter(self.validation_df[feature], self.val_residuals, alpha = 0.5)
            axs[ax_index, 0].set_xlabel(feature)
            axs[ax_index, 0].set_ylabel("Residuals")
            axs[ax_index, 0].set_title("fea_dept_number vs Predictions")
            axs[ax_index, 0].grid(True)

            # Feature vs Prediction
            axs[ax_index, 1].scatter(self.validation_df[feature], self.val_predictions, alpha = 0.5)
            axs[ax_index, 1].set_xlabel(feature)
            axs[ax_index, 1].set_ylabel("Predictions")
            axs[ax_index, 1].set_title("fea_dept_number vs Predictions")
            axs[ax_index, 1].grid(True)
            plt.tight_layout()

            plot_path = os.path.join(self.run_directory, f"feature_importance_residuals.png")
            plt.savefig(plot_path)
            #if not self.plot_residuals_paths:
            self._plot_importance_path = plot_path
    

        
