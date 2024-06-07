from src.uc_api.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.uc_api.utils.common import read_yaml, create_directories
from src.uc_api.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    SentimentModelTrainerConfig,
    SentimentPredictionConfig,
)


class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
        ):

        # Reading yaml files for config and params    
        self.config = read_yaml(config_filepath) 
        self.params = read_yaml(params_filepath)  

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Data Ingestion configurations from yaml file
        config = self.config.data_ingestion

        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments   

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            # Data Ingestion configurations
            root_dir=config.root_dir,
            unzip_dir=config.unzip_dir,
            local_data_file=config.local_data_file,

            # Common Arguments Parameters
            data_split_type=params_common.data_split_type,
            data_split_ratio=params_common.data_split_ratio,

            # Use case specific parameters   
            source_URL=config.source_URL,
            outsource_file=config.outsource_file, 
            feature_column_name=params.feature_column_name,
            label_column_name = params.label_column_name,

        )

        return data_ingestion_config
    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        # Data Validation configurations from yaml file
        config = self.config.data_validation
        
        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            # Data Validation configurations
            root_dir=config.root_dir,
            status_file=config.status_file,
            
            # Common Arguments Parameters
            data_split_type=params_common.data_split_type,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        # Data Transformation configurations from yaml file
        config = self.config.data_transformation

        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments        

        create_directories([config.root_dir])

        # Populating the data class 
        data_transformation_config = DataTransformationConfig(
            # Data Transformation configurations
            root_dir=config.root_dir,
            dataset_path=config.dataset_path,
            tokenizer_name=config.tokenizer_name,

            # Common Arguments Parameters
            data_split_type=params_common.data_split_type,

            # Use case specific(Sentiment) parameters           
            feature_column_name=params.feature_column_name,
            label_column_name=params.label_column_name,
            sentiment_class_dict = params.sentiment_class_dict,
            batch_size=params.batch_size,
            max_length=params.max_length,
        )

        return data_transformation_config
    

    def get_sentiment_model_trainer_config(self) -> SentimentModelTrainerConfig:
        # Data Transformation configurations from yaml file
        config_prev = self.config.data_transformation

        # Model Trainer and Evaluation configurations
        config = self.config.sentiment_model_trainer_and_evaluation

        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments        

        create_directories([config.root_dir])

        sentiment_model_trainer_config = SentimentModelTrainerConfig(
            # Sentiment Data Transformation configurations            
            transformation_root_dir=config_prev.root_dir, 
            tokenizer_name=config_prev.tokenizer_name,           

            # Sentiment Model Trainer configurations
            root_dir=config.root_dir,
            pre_trained_model_name=config.pre_trained_model_name,           
            model_ckpt=config.model_ckpt,

            # Common Parameters
            data_split_type=params_common.data_split_type,

            # Parameters
            sentiment_class_dict=params.sentiment_class_dict,            
            num_train_epochs=params.num_train_epochs,
            dropout_parameter=params.dropout_parameter,
            max_length=params.max_length,
        )

        return sentiment_model_trainer_config
