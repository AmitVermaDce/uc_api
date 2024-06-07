import os
import kaggle
import zipfile
from pathlib import Path
import urllib.request as request
from datasets import load_dataset
from src.uc_api.logging import logger
from src.uc_api.utils.common import get_size
from src.uc_api.entity import DataIngestionConfig



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self, source=None):    
        """
        source: Path
        Download data from kaggle in zip format
        Download data from URL in zip format
        Function returns None
        """                   
        if not os.path.exists(self.config.local_data_file):
            # Downlad data from kaggle in zip form
            if source == "kaggle":
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(self.config.source_URL, self.config.root_dir, unzip=False)
                os.rename(os.path.join(self.config.root_dir, self.config.outsource_file), self.config.local_data_file)
                logger.info(f"Dataset downloaded from {source}")
            # Downlad data from URL in zip form
            else:
                filename, headers = request.urlretrieve(
                    url=self.config.source_URL, filename=self.config.local_data_file
                )
                logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

    def convert_csv_to_train_and_test_datasets(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        try:
            suffix = "csv"
            filenames = os.listdir(self.config.unzip_dir)
            selected_file = [
                filename for filename in filenames if filename.endswith(suffix) and filename.__contains__("train")
            ][0]

            dataset = load_dataset(
                suffix, 
                data_files=os.path.join(self.config.unzip_dir, selected_file),
                encoding='unicode_escape',
                
            )["train"].train_test_split(test_size=self.config.data_split_ratio, seed=42)
            dataset.save_to_disk(os.path.join(self.config.root_dir, "processed"))
            logger.info("Successfully created arrow dataset")
        except Exception as e:
            logger.info("Error creating arrow dataset")
            raise e