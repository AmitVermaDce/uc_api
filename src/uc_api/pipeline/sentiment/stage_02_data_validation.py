from src.uc_api.config.configuration import ConfigurationManager
from uc_api.components.sentiment.data_validation import (
    DataValidation,
)


class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_sentiment_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_files_exist()
