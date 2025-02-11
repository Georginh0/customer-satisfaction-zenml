class ModelNameConfig:
    """Model Configurations"""

    def __init__(
        self, model_name: str = "lightgbm", fine_tuning: bool = False
    ):
        self.model_name = model_name
        self.fine_tuning = fine_tuning
