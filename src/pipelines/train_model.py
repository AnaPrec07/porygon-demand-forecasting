import logging
import json
from pathlib import Path
from datetime import datetime

from src.core.clients.bigquery import BigQueryClient
from src.core.config_loader import ConfigLoader
from src.core.models.base_model import BaseTrainer

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
    training_df = bq_client.load_table(
        table_name=config_loader.training_table_name,
        fields=fields
    )

    # Prepare features and target
    X = training_df[config_loader.features_list]
    y = training_df[config_loader.target_col]
    
    # Train-test split (time-based)
    split_idx = int(training_df.shape[0] * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 01. Train model
    logger.info("Training model...")
    model_trainer.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test
    )
    

    
    # 7. Create visualizations (optional)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Actual vs Predicted plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Actual vs Predicted scatter
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_test.values - y_pred.values
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
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
        if feature_importance is not None and len(feature_importance) > 0:
            top_features = feature_importance.head(15)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].set_title('Top 15 Feature Importance')
            axes[1, 1].invert_yaxis()
            axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plots_path = run_dir / "evaluation_plots.png"
        plt.savefig(plots_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Evaluation plots saved: {plots_path}")
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")
    


if __name__ == "__main__":
    main()
