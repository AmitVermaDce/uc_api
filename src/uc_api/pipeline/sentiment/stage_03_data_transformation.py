from src.uc_api.config.configuration import ConfigurationManager
from uc_api.components.sentiment.data_transformation import (
    DataTransformation,
)


class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_sentiment_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config,)
        data_transformation.transform_and_save_to_file()
