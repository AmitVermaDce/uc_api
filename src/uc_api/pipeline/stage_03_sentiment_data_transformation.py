from src.uc_api.config.configuration import ConfigurationManager
from src.uc_api.components.data_transformation_sentiment import (
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
