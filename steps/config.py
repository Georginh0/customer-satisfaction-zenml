from zenml.steps import BasedParameters


class ModelNameConfig(BasedParameters):
    """model config"""

    model_name: str = "LinearRegression"
