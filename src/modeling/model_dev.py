import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression


class BaseModel(ABC):
    """
    Abstract class for model strategies.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model.

        Args:
            X_train : training data
            y_train : training labels
        Returns:
            Trained model
        """
        pass


class LinearRegressionModel(BaseModel):
    """
    Strategy for Linear Regression.
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the Linear Regression model.

        Args:
            X_train : training data
            y_train : training labels
        Returns:
            Trained model
        """
        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model Trained")
            return reg  # ✅ Fixed: return is inside the function
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
