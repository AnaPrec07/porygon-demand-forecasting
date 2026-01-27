"""Config Loader."""
from pathlib import Path
import yaml
import os
from pandas import DataFrame
import pandas as pd
from google.oauth2 import service_account

class ConfigLoader:
    """Configuration Loader Class."""
    def __init__(self):
        """Initialize Config Loader."""
        self.config_directory_path = os.path.join(Path(__file__).parent.parent, "config")
        self.set_full_config()
        self.get_sa_credentials()

    def _load_config_yaml(self):
        with open(os.path.join(self.config_directory_path, "config.yaml"), "r") as f:
            self.config_yaml = yaml.safe_load(f)

    def set_full_config(self):
        """Set attributes on target object from config key.
        Args:
            target_obj: Object to set attributes on
            key: Config key containing attribute dict
        """
        # load config yaml.
        self._load_config_yaml()
        
        # set config yaml configurations.
        for _, config_dict in self.config_yaml.items():
            for attr_name, attr_value in config_dict.items():
                setattr(self, attr_name, attr_value)

        # Set training fields.
        self.inventory_sim_required_fields =[
            "fea_item_store_price_avg",
            "fea_item_monthly_sales_roll_std_3_months",
            "fea_item_monthly_sales"
        ]
        self.required_fields = [self.target_col, self.date_column, self.benchmark_col] + self.id_columns + self.inventory_sim_required_fields
    
    def get_sa_credentials(self):
        """Get service account credentials from json file."""
        sa_credentials_path = os.path.join(self.config_directory_path, "credentials", "service_account.json")
        self.credentials = service_account.Credentials.from_service_account_file(
                sa_credentials_path,
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/bigquery",
                ]
        )


    def extract_config_dataframe(self) -> DataFrame:
        """Get all configurations as a pandas DataFrame."""

        records = []
        for key, config_dict in self.config_yaml.items():
            for param, value in config_dict.items():
                records.append({
                    "parameter": param,
                    "value": value,
                    "source": key
                })
        return pd.DataFrame(records)

