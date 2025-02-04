import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract class for evaluation strategies
    """

    @abstractmethod
    def calculate_scores(self, y: np.ndarray, y_pred: np.ndarray):
        """
        calculate score for the model

          Args:

              X_test : test data
              y_test : test labels
          Returns:
              None
        """
        pass


class MSE(Evaluation):
    """
    Evaluation Strategy for Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculate score for the model

          Args:

              y_true : true labels
              y_pred : predicted labels
          Returns:
              None
        """
        try:
            logging.info("Calculating Mean Squared Error")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("Mean Squared Error:{}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE:{}".format(e))
            raise e


class R2(Evaluation):
    """
    Evaluation Strategy for R2 Score
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculate score for the model

          Args:

              y_true : true labels
              y_pred : predicted labels
          Returns:
              None
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score:{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score:{}".format(e))
            raise e


class RMSE(Evaluation):
    """
    Evaluation Strategy for Root Mean Squared Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        calculate score for the model

          Args:

              y_true : true labels
              y_pred : predicted labels
          Returns:
              None
        """
        try:
            logging.info("Calculating Root Mean Squared Error")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("Root Mean Squared Error:{}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE:{}".format(e))
            raise e
