from uc_api.pipeline.sentiment.stage_01_data_ingestion import (
    DataIngestionPipeline,
)
from uc_api.pipeline.sentiment.stage_02_data_validation import (
    DataValidationPipeline,
)
from src.uc_api.pipeline.sentiment.stage_03_data_transformation import (
    DataTransformationPipeline,
)
from uc_api.pipeline.sentiment.stage_04_model_trainer_and_evaluate import (
    ModelTrainerAndEvaluationPipeline,
)
from uc_api.pipeline.sentiment.stage_05_model_test import (
    ModelTestingPipeline,
)
from src.uc_api.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Sentiment Data Transformation stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer & Evaluation stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerAndEvaluationPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Testing stage"
try:
    logger.info(f">>>>>> stage: {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTestingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage: {STAGE_NAME} completed <<<<<<\n\nx===============================x")
except Exception as e:
    logger.exception(e)
    raise e