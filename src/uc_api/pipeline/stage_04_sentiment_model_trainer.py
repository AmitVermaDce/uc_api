from src.uc_api.config.configuration import ConfigurationManager
from src.uc_api.components.sentiment_model_trainer import SentimentModelTrainer


class SentimentModelTrainerAndEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_sentiment_model_trainer_config()
        model_trainer_config = SentimentModelTrainer(
            config=model_trainer_config,
        )
        model_trainer_config.train_and_evaluate_sentiment_model()
