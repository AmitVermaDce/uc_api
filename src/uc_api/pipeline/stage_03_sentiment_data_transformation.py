from src.uc_api.config.configuration import ConfigurationManager
from uc_api.components.sentiment_data_transformation import (
    SentimentDataTransformation,
)


class SentimentDataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = SentimentDataTransformation(config=data_transformation_config,)
        data_transformation.transform_and_save_to_file()
