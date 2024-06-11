from src.uc_api.config.configuration import ConfigurationManager
from src.uc_api.components.sentiment.model_testing import (
    ModelTesting,
)


class ModelTestingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_testing_config = config.get_sentiment_model_testing_config()
        model_testing_config = ModelTesting(
            config=model_testing_config,
        )
        model_testing_config.test_sentiment_model()
