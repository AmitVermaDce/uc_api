from dataclasses import dataclass
from pathlib import Path

# <=============== Sentiment ===============>
@dataclass(frozen=True)
class SentimentDataIngestionConfig:
    # Common Configurations
    root_dir: Path   
    unzip_dir: Path   
    local_data_file: Path

    # Common Parameters
    data_split_type: list
    data_split_ratio: float   

    # Parameters
    source_URL: Path  
    outsource_file: Path
    feature_column_name: str
    label_column_name: str


@dataclass(frozen=True)
class SentimentDataValidationConfig:
    # Common Configurations
    root_dir: Path
    status_file: str
    
    # Common Parameters
    data_split_type: list


@dataclass(frozen=True)
class SentimentDataTransformationConfig:
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
class SentimentModelTrainerEvaluationConfig:
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
class SentimentModelTestingConfig:
    
    # Use case specific Configurations
    test_dataset_path:str
    pre_trained_model_name:str           
    model_ckpt:Path    

    # Use case specific Parameters
    sentiment_class_dict: dict       
    max_length: int
    dropout_parameter: float


@dataclass(frozen=True)
class SentimentPredictionConfig:
    tokenizer_name: str
    pre_trained_model_name: str
    model_ckpt: Path
    sentiment_class_dict: dict
    max_length: int
    dropout_parameter: float


# <=============== Summarizer ===============>
@dataclass(frozen=True)
class SummarizerDataIngestionConfig:
    # Common Configurations
    root_dir: Path   
    unzip_dir: Path   
    local_data_file: Path

    # Common Parameters
    data_split_type: list
    data_split_ratio: float   

    # Parameters
    source_URL: Path  
    outsource_file: Path
    feature_column_name: str
    label_column_name: str

@dataclass(frozen=True)
class SummarizerDataValidationConfig:
    # Common Configurations
    root_dir: Path
    status_file: str
    
    # Common Parameters
    data_split_type: list