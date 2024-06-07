import os
from src.uc_api.entity import DataValidationConfig
from src.uc_api.logging import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            # Check for directories based on data split type example [train, test]
            validation_status = True
            path = os.path.join("artifacts", "data_ingestion", "processed")
            dataset_dirs = [f.name for f in os.scandir(path) if f.is_dir()]
            for dataset_type in dataset_dirs:
                if dataset_type not in self.config.data_split_type:
                    validation_status = False
            with open(self.config.status_file, "w") as f:        
                f.write(f"Validation status: {validation_status}")                        
                logger.info(f"Validation status: {validation_status}")
            return validation_status

        except Exception as e:
            raise e