import os
from textSummarizer.entity import DataValidationConfig
from textSummarizer.logging import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True
            all_files = os.listdir(os.path.join("artifacts", "data_ingestion", "summarizer-data","samsum_dataset"))

            for file in self.config.ALL_REQUIRED_FILES:
                if file not in all_files:
                    validation_status = False
                    logger.warning(f"Missing file: {file}")
                    break

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Validation status written to {self.config.STATUS_FILE}")
            return validation_status

        except Exception as e:
            logger.exception("Error during data validation")
            raise e
