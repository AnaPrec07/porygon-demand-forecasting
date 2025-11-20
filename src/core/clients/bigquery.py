"""BigQueryClient Class."""

import logging
from typing import Optional, List, Dict, Any
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core import exceptions
import yaml
import argparse
from pathlib import Path

from src.core.config_loader import ConfigLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryClient:
    """Utility class for BigQuery operations."""
    
    def __init__(self):
        """Initialize BigQuery client."""
        # Set configurations.
        self.config = ConfigLoader()

        # Set BigQuery client.
        self._set_client()
    
    def _set_client(self):
        """Set BigQuery client."""
        self._client = bigquery.Client(
                project=self.config.project_id,
                credentials=self.config.credentials
        )
        logger.info(f"BigQuery client initialized for project: {self.config.project_id}")

    def load_table(self, table_name: str, fields: Optional[str] = "*", sample=True) -> pd.DataFrame:
        """Load entire table into a DataFrame."""
        if sample:
            query = f"SELECT {fields} FROM `{self.config.project_id}.{self.config.dataset_id}.{table_name}` WHERE ctx_dept_id = 'FOODS_3'  AND ctx_store_id = 'CA_1' "
        else:
            query = f"SELECT {fields} FROM `{self.config.project_id}.{self.config.dataset_id}.{table_name}`"
        df = self._client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} rows from table: {table_name}")
        return df