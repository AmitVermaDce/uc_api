from src.uc_api.config.configuration import ConfigurationManager
from uc_api.components.sentiment_data_ingestion import DataIngestion


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file(source="kaggle")
        data_ingestion.extract_zip_file()
        data_ingestion.convert_csv_to_train_and_test_datasets()
