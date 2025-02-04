import logging
from zenml import step
import pandas as pd
from src.modeling.evalution import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple


@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2"],
    Annotated[float, "rsme"],
]:
    """
    Evaluates the model on the Ingested Data


    Args:
        df : the ingested data
    """
    try:
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, predictions)

        return r2, mse, rmse
    except Exception as e:
        logging.error("Error in evaluating model:{}".format(e))
        raise e
