# import logging
import pandas as pd
from zenml import step


@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train the Model on the Ingest Data

    Args:
        df: Ingest Data
    """
    pass
