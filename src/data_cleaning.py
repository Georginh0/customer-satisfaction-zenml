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


class DataPreProcessStrategy:
    """
    Strategy for data preprocessing
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle data preprocessing
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            col_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(col_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("error in data preprocessing data :{}".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for data splitting into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.DataFrame]:
        """
        Handle data splitting into train and test
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("error in data splitting data :{}".format(e))
            raise e


class DataCleaning:
    """
    class for cleaning  data which processes the data and divides it into train and testt
    """
    def __init__(self,data:pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]
        """
        Handle data cleaning
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("error in data cleaning data :{}".format(e))  
            raise e
        
