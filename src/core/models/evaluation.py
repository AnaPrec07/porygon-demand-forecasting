"""Model Evaluation."""

# Import required libraries

from src.core.config_loader import ConfigLoader
from src.core.clients.bigquery import BigQueryClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error, median_absolute_error, mean_pinball_loss

bq_client = BigQueryClient()
config_loader = ConfigLoader()

def error_for_group(group, model):
    group["actuals"] = group[config_loader.target_col]
    group["benchmark"] = group[config_loader.benchmark_col]
    group["predictions"] = model.predict(group[model.features], True)

    return pd.Series({
        "pred_mae": mean_absolute_error(group["actuals"], group["p" \
        "" \
        "" \
        "redictions"]),
        "bench_mae": mean_absolute_error(group["actuals"], group["benchmark"]),
        "diff_mae": (
            mean_absolute_error(group["actuals"], group["predictions"])
            - mean_absolute_error(group["actuals"], group["benchmark"])
        ),
        "pred_rsme": root_mean_squared_error(group["actuals"], group["predictions"]),
        "bench_rsme": root_mean_squared_error(group["actuals"], group["benchmark"]),
        "diff_rsme": (
            root_mean_squared_error(group["actuals"], group["predictions"])
            - root_mean_squared_error(group["actuals"], group["benchmark"])
        ),
        "pred_mdae": median_absolute_error(group["actuals"], group["predictions"]),
        "bench_mdae": median_absolute_error(group["actuals"], group["benchmark"]),
        "diff_mdae": (
            median_absolute_error(group["actuals"], group["predictions"])
            - median_absolute_error(group["actuals"], group["benchmark"])
        ),
        "pred_mape": mean_absolute_percentage_error(group["actuals"], group["predictions"]),
        "bench_mape": mean_absolute_percentage_error(group["actuals"], group["benchmark"]),
        "diff_mape": (
            mean_absolute_percentage_error(group["actuals"], group["predictions"])
            - mean_absolute_percentage_error(group["actuals"], group["benchmark"])
        ),
        "pred_mdape": (abs(group["predictions"]-group["actuals"])/group["actuals"]).quantile(.50),
        "bench_mdape": (abs(group["benchmark"]-group["actuals"])/group["actuals"]).quantile(.50),
        "diff_mdape": ((abs(group["predictions"]-group["actuals"])/group["actuals"])).quantile(.50)
            - (abs(group["benchmark"]-group["actuals"])/group["actuals"]).quantile(.50)
        })

def retrieve_error_per_group(df, groupby_col, model):
    df = df.copy()

    return df.groupby(groupby_col).apply(lambda group: error_for_group(group, model))

