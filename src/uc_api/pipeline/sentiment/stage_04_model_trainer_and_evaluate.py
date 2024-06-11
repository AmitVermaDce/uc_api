from src.uc_api.config.configuration import ConfigurationManager
from src.uc_api.components.sentiment.model_trainer import (
    ModelTrainerAndEvaluation,
)


class ModelTrainerAndEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_sentiment_model_trainer_and_evaluation_config()
        model_trainer_config = ModelTrainerAndEvaluation(
            config=model_trainer_config,
        )
        model_trainer_config.train_and_evaluate_sentiment_model()
