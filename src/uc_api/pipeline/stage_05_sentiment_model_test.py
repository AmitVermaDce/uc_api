from src.uc_api.config.configuration import ConfigurationManager
from src.uc_api.components.sentiment_model_testing import SentimentModelTesting


class SentimentModelTestingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_testing_config = config.get_sentiment_model_testing_config()
        model_testing_config = SentimentModelTesting(
            config=model_testing_config,
        )
        model_testing_config.test_sentiment_model()
