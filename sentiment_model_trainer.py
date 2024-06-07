from src.uc_api.pipeline.stage_01_data_ingestion_sentiment import (
    DataIngestionTrainingPipeline,
)
from src.uc_api.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from src.uc_api.pipeline.stage_03_sentiment_data_transformation import (
    SentimentDataTransformationPipeline,
)
from src.uc_api.pipeline.stage_04_sentiment_model_trainer import (
    SentimentModelTrainerAndEvaluationPipeline,
)
from src.uc_api.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Sentiment Data Transformation stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    data_transformation = SentimentDataTransformationPipeline()
    data_transformation.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer & Evaluation stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    model_trainer = SentimentModelTrainerAndEvaluationPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e