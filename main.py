from textSummarizer.logging import logger

from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>>> Stage: {STAGE_NAME} started <<<<<<<<")
    
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logger.info(f">>>>>>> Stage: {STAGE_NAME} completed <<<<<<<<\n{'x'*30}\n")

except Exception as e:
    logger.exception(e)
    raise e
