from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    # Common Configurations
    root_dir: Path   
    unzip_dir: Path   
    local_data_file: Path

    # Common Parameters
    data_split_ratio: float   

    # Parameters
    source_URL: Path  
    outsource_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    # Common Configurations
    root_dir: Path
    status_file: str
    
    # Common Parameters
    data_split_type: list


@dataclass(frozen=True)
class DataTransformationConfig:
    # Common Configurations
    root_dir: Path
    dataset_path: Path
    tokenizer_name: str

    # Common Parameters
    data_split_type: list 

    # Parameters
    feature_column_name: str 
    label_column_name: str 
    sentiment_class_dict: dict 
    batch_size: int
    max_length: int


@dataclass(frozen=True)
class SentimentModelTrainerConfig:
    # Common Parameters
    data_split_type: list 

    # Use case specific Configurations
    transformation_root_dir: Path   
    tokenizer_name: str
    root_dir: Path 
    pre_trained_model_name: str
    model_ckpt: Path

    # Use case specific Parameters
    sentiment_class_dict: dict
    num_train_epochs: int    
    dropout_parameter: float        
    max_length: int


@dataclass(frozen=True)
class SentimentPredictionConfig:
    tokenizer_name: str
    pre_trained_model_name: str
    model_ckpt: Path
    sentiment_class_dict: dict
    max_length: int
    dropout_parameter: float