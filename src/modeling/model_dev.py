import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from .config import ModelNameConfig


class model(ABC):
    """
    Abstract class for model strategies
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        trains the model

        Args:
            X_train : training data
            y_train : training labels
        Returns:
            None
        """
        pass


class LinearRegressionModel(model):
    """
    Strategy for Linear Regression
    """

    def train(self, X_train, y_train, **kwargs):
        """
        trains the model

        Args:
            X_train : training data
            y_train : training labels
        Returns:
            None
        """

    try:
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        logging.info("Model Trained")
        return reg
    except Exception as e:
        logging.error("Error in training model:{}".format(e))
        raise e
