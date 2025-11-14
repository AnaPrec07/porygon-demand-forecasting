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

    def load_table(self, table_name: str, fields: Optional[str] = "*") -> pd.DataFrame:
        """Load entire table into a DataFrame."""
        query = f"SELECT {fields} FROM `{self.config.project_id}.{self.config.dataset_id}.{table_name}`"
        df = self._client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} rows from table: {table_name}")
        return df





##    def read_table(
##        self,
##        table_id: str,
##        dataset_id: Optional[str] = None,
##        limit: Optional[int] = None,
##        columns: Optional[List[str]] = None,
##        where_clause: Optional[str] = None
##    ) -> pd.DataFrame:
##        """
##        Read data from BigQuery table.
##        
##        Args:
##            table_id: Table name
##            dataset_id: Optional dataset ID (if not in table_id)
##            limit: Optional row limit
##            columns: Optional list of columns to select
##            where_clause: Optional WHERE clause (without WHERE keyword)
##            
##        Returns:
##            DataFrame with query results
##        """
##        try:
##            # Construct full table reference
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            elif '.' not in table_id:
##                raise ValueError("Either provide dataset_id or use format 'dataset.table'")
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            # Build query
##            select_clause = ", ".join(columns) if columns else "*"
##            query = f"SELECT {select_clause} FROM `{full_table_id}`"
##            
##            if where_clause:
##                query += f" WHERE {where_clause}"
##            
##            if limit:
##                query += f" LIMIT {limit}"
##            
##            logger.info(f"Executing query: {query}")
##            df = self.client.query(query).to_dataframe()
##            logger.info(f"Read {len(df)} rows from {full_table_id}")
##            
##            return df
##            
##        except exceptions.NotFound:
##            logger.error(f"Table not found: {full_table_id}")
##            raise
##        except Exception as e:
##            logger.error(f"Error reading table: {str(e)}")
##            raise
##    
##    def load_table(
##        self,
##        df: pd.DataFrame,
##        table_id: str,
##        dataset_id: Optional[str] = None,
##        write_disposition: str = "WRITE_TRUNCATE",
##        schema: Optional[List[bigquery.SchemaField]] = None
##    ) -> bigquery.LoadJob:
##        """
##        Load DataFrame to BigQuery table.
##        
##        Args:
##            df: DataFrame to load
##            table_id: Target table name
##            dataset_id: Optional dataset ID
##            write_disposition: Write mode (WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)
##            schema: Optional schema definition
##            
##        Returns:
##            Load job object
##        """
##        try:
##            # Construct full table reference
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            elif '.' not in table_id:
##                raise ValueError("Either provide dataset_id or use format 'dataset.table'")
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            # Configure load job
##            job_config = bigquery.LoadJobConfig(
##                write_disposition=write_disposition,
##                schema=schema
##            )
##            
##            logger.info(f"Loading {len(df)} rows to {full_table_id}")
##            job = self.client.load_table_from_dataframe(
##                df,
##                full_table_id,
##                job_config=job_config
##            )
##            
##            job.result()  # Wait for completion
##            logger.info(f"Successfully loaded data to {full_table_id}")
##            
##            return job
##            
##        except Exception as e:
##            logger.error(f"Error loading table: {str(e)}")
##            raise
##    
##    def execute_query(
##        self,
##        query: str,
##        job_config: Optional[bigquery.QueryJobConfig] = None
##    ) -> pd.DataFrame:
##        """
##        Execute a SQL query and return results.
##        
##        Args:
##            query: SQL query string
##            job_config: Optional query job configuration
##            
##        Returns:
##            DataFrame with query results
##        """
##        try:
##            logger.info("Executing custom query")
##            query_job = self.client.query(query, job_config=job_config)
##            df = query_job.to_dataframe()
##            logger.info(f"Query returned {len(df)} rows")
##            return df
##            
##        except Exception as e:
##            logger.error(f"Error executing query: {str(e)}")
##            raise
##    
##    def table_exists(
##        self,
##        table_id: str,
##        dataset_id: Optional[str] = None
##    ) -> bool:
##        """
##        Check if a table exists.
##        
##        Args:
##            table_id: Table name
##            dataset_id: Optional dataset ID
##            
##        Returns:
##            True if table exists, False otherwise
##        """
##        try:
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            self.client.get_table(full_table_id)
##            return True
##            
##        except exceptions.NotFound:
##            return False
##    
##    def create_table(
##        self,
##        table_id: str,
##        schema: List[bigquery.SchemaField],
##        dataset_id: Optional[str] = None,
##        clustering_fields: Optional[List[str]] = None,
##        partitioning_field: Optional[str] = None
##    ) -> bigquery.Table:
##        """
##        Create a new BigQuery table.
##        
##        Args:
##            table_id: Table name
##            schema: Table schema
##            dataset_id: Optional dataset ID
##            clustering_fields: Optional clustering fields
##            partitioning_field: Optional partitioning field
##            
##        Returns:
##            Created table object
##        """
##        try:
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            table = bigquery.Table(full_table_id, schema=schema)
##            
##            if clustering_fields:
##                table.clustering_fields = clustering_fields
##            
##            if partitioning_field:
##                table.time_partitioning = bigquery.TimePartitioning(
##                    type_=bigquery.TimePartitioningType.DAY,
##                    field=partitioning_field
##                )
##            
##            table = self.client.create_table(table)
##            logger.info(f"Created table {full_table_id}")
##            
##            return table
##            
##        except Exception as e:
##            logger.error(f"Error creating table: {str(e)}")
##            raise
##    
##    def delete_table(
##        self,
##        table_id: str,
##        dataset_id: Optional[str] = None,
##        not_found_ok: bool = True
##    ):
##        """
##        Delete a BigQuery table.
##        
##        Args:
##            table_id: Table name
##            dataset_id: Optional dataset ID
##            not_found_ok: If True, don't raise error if table doesn't exist
##        """
##        try:
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            self.client.delete_table(full_table_id, not_found_ok=not_found_ok)
##            logger.info(f"Deleted table {full_table_id}")
##            
##        except Exception as e:
##            logger.error(f"Error deleting table: {str(e)}")
##            raise
##    
##    def get_table_schema(
##        self,
##        table_id: str,
##        dataset_id: Optional[str] = None
##    ) -> List[bigquery.SchemaField]:
##        """
##        Get schema of a table.
##        
##        Args:
##            table_id: Table name
##            dataset_id: Optional dataset ID
##            
##        Returns:
##            List of schema fields
##        """
##        try:
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            table = self.client.get_table(full_table_id)
##            return table.schema
##            
##        except Exception as e:
##            logger.error(f"Error getting table schema: {str(e)}")
##            raise
##    
##    def list_tables(self, dataset_id: str) -> List[str]:
##        """
##        List all tables in a dataset.
##        
##        Args:
##            dataset_id: Dataset ID
##            
##        Returns:
##            List of table names
##        """
##        try:
##            dataset_ref = f"{self.project_id}.{dataset_id}"
##            tables = self.client.list_tables(dataset_ref)
##            table_names = [table.table_id for table in tables]
##            logger.info(f"Found {len(table_names)} tables in {dataset_id}")
##            return table_names
##            
##        except Exception as e:
##            logger.error(f"Error listing tables: {str(e)}")
##            raise
##    
##    def get_table_info(
##        self,
##        table_id: str,
##        dataset_id: Optional[str] = None
##    ) -> Dict[str, Any]:
##        """
##        Get detailed information about a table.
##        
##        Args:
##            table_id: Table name
##            dataset_id: Optional dataset ID
##            
##        Returns:
##            Dictionary with table information
##        """
##        try:
##            if dataset_id and '.' not in table_id:
##                full_table_id = f"{self.project_id}.{dataset_id}.{table_id}"
##            else:
##                full_table_id = f"{self.project_id}.{table_id}" if table_id.count('.') == 1 else table_id
##            
##            table = self.client.get_table(full_table_id)
##            
##            return {
##                'table_id': table.table_id,
##                'dataset_id': table.dataset_id,
##                'project': table.project,
##                'num_rows': table.num_rows,
##                'num_bytes': table.num_bytes,
##                'created': table.created,
##                'modified': table.modified,
##                'schema': [{'name': field.name, 'type': field.field_type} for field in table.schema],
##                'description': table.description
##            }
##            
##        except Exception as e:
##            logger.error(f"Error getting table info: {str(e)}")
##            raise
##
##
### Convenience function for quick access
##def get_bq_client(project_id: str, credentials_path: Optional[str] = None) -> BigQueryClient:
##    """
##    Get a BigQuery client instance.
##    
##    Args:
##        project_id: GCP project ID
##        credentials_path: Optional path to service account JSON
##        
##    Returns:
##        BigQueryClient instance
##    """
##    return BigQueryClient(project_id, credentials_path)