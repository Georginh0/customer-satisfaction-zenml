import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class for data cleaning strategies
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for data preprocessing
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle data preprocessing
        """
        try:
            # Ensure column names are in standard format
            data.columns = data.columns.str.strip().str.lower()

            # Define columns to drop and ensure they exist before dropping
            columns_to_drop = [
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
                "order_purchase_timestamp",
            ]
            data.drop(
                columns=[col for col in columns_to_drop if col in data.columns],
                inplace=True,
                errors="ignore",
            )

            # Fill missing values in numerical columns efficiently
            numeric_cols = [
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
            for col in numeric_cols:
                if col in data.columns:
                    median_value = data[col].median()
                    data[col] = data[col].fillna(median_value)

            # Fill missing values in categorical columns efficiently
            if "review_comment_message" in data.columns:
                data["review_comment_message"] = data["review_comment_message"].fillna(
                    "No review"
                )

            # Drop specific columns if they exist
            drop_cols = ["customer_zip_code_prefix", "order_item_id"]
            data.drop(
                columns=[col for col in drop_cols if col in data.columns],
                inplace=True,
                errors="ignore",
            )

            # Encode categorical features efficiently
            categorical_cols = data.select_dtypes(include=["object"]).columns
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

            return data

        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for data splitting into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data splitting into train and test
        """
        try:
            if "review_score" not in data.columns:
                raise ValueError("Column 'review_score' is missing in the dataset")

            # Convert numeric columns to float32 to save memory
            for col in data.select_dtypes(include=["number"]).columns:
                data[col] = data[col].astype(np.float32)

            X = data.drop("review_score", axis=1)
            y = data["review_score"].astype(
                np.int8
            )  # Use int8 if values are small integers

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise e


class DataCleaning:
    """
    Class for cleaning data, which processes the data and divides it into train and test sets
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data cleaning
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in data cleaning: {e}")
            raise e
