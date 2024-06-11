from src.uc_api.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.uc_api.utils.common import read_yaml, create_directories
from src.uc_api.entity import (
    SentimentDataIngestionConfig,
    SentimentDataValidationConfig,
    SentimentDataTransformationConfig,
    SentimentModelTrainerEvaluationConfig,
    SentimentModelTestingConfig,
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

    def get_sentiment_data_ingestion_config(self) -> SentimentDataIngestionConfig:
        # Data Ingestion configurations from yaml file
        config = self.config.sentiment_data_ingestion

        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments   

        create_directories([config.root_dir])

        sentiment_data_ingestion_config = SentimentDataIngestionConfig(
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

        return sentiment_data_ingestion_config
    
    
    def get_sentiment_data_validation_config(self) -> SentimentDataValidationConfig:
        # Data Validation configurations from yaml file
        config = self.config.sentiment_data_validation
        
        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        create_directories([config.root_dir])

        sentiment_data_validation_config = SentimentDataValidationConfig(
            # Data Validation configurations
            root_dir=config.root_dir,
            status_file=config.status_file,
            
            # Common Arguments Parameters
            data_split_type=params_common.data_split_type,
        )

        return sentiment_data_validation_config
    

    def get_sentiment_data_transformation_config(self) -> SentimentDataTransformationConfig:
        # Data Transformation configurations from yaml file
        config = self.config.sentiment_data_transformation

        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments        

        create_directories([config.root_dir])

        # Populating the data class 
        sentiment_data_transformation_config = SentimentDataTransformationConfig(
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

        return sentiment_data_transformation_config
    

    def get_sentiment_model_trainer_and_evaluation_config(self) -> SentimentModelTrainerEvaluationConfig:
        # Data Transformation configurations from yaml file
        config_prev = self.config.sentiment_data_transformation

        # Model Trainer and Evaluation configurations
        config = self.config.sentiment_model_trainer_and_evaluation

        # Common Arguments Parameters from yaml file
        params_common = self.params.CommonArguments

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments        

        create_directories([config.root_dir])

        sentiment_model_trainer_evaluation_config = SentimentModelTrainerEvaluationConfig(
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

        return sentiment_model_trainer_evaluation_config
    

    def get_sentiment_model_testing_config(self) -> SentimentModelTestingConfig:
        # Data Transformation configurations from yaml file
        config_prev_transformation = self.config.sentiment_data_transformation

        # Model Trainer and Evaluation configurations from yaml file
        config_prev_model_trainer = self.config.sentiment_model_trainer_and_evaluation

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments  

        sentiment_model_testing_config = SentimentModelTestingConfig(

            # Sentiment Data Ingestion configurations
            test_dataset_path=config_prev_transformation.root_dir,

            # Sentiment Model Trainer configurations
            pre_trained_model_name=config_prev_model_trainer.pre_trained_model_name,           
            model_ckpt=config_prev_model_trainer.model_ckpt,

            # Sentiment Parameters
            sentiment_class_dict=params.sentiment_class_dict,  
            max_length=params.max_length,
            dropout_parameter=params.dropout_parameter,
        )

        return sentiment_model_testing_config


    def get_sentiment_prediction_config(self) -> SentimentPredictionConfig:

        config_prev_transformation = self.config.sentiment_data_transformation

        # Model Trainer and Evaluation configurations from yaml file
        config_prev_model_trainer = self.config.sentiment_model_trainer_and_evaluation

        # Use case specific(Sentiment) parameters from yaml file
        params = self.params.SentimentArguments  

        sentiment_model_prediction_config = SentimentPredictionConfig(

            # Sentiment Transformation configurations            
            tokenizer_name=config_prev_transformation.tokenizer_name,

            # Sentiment Model Trainer configurations           
            pre_trained_model_name=config_prev_model_trainer.pre_trained_model_name,
            model_ckpt=config_prev_model_trainer.model_ckpt,

            # Sentiment Parameters
            sentiment_class_dict=params.sentiment_class_dict,  
            max_length=params.max_length,
            dropout_parameter=params.dropout_parameter,
        )

        return sentiment_model_prediction_config