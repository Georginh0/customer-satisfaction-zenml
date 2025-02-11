import logging
import pandas as pd
from zenml import step
from src.modeling.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin


@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_name: str = "LinearRegression",  # Explicitly pass the model name
) -> RegressorMixin:
    """
    Train the Model on the Ingest Data.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_name: Name of the model to train (default: LinearRegression)

    Returns:
        Trained model
    """
    try:
        model = None
        if model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {model_name} not found")
    except Exception as e:
        logging.error(f"Error in training model: {e}")
        raise e
