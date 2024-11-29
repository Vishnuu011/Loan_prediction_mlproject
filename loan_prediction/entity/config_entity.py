from datetime import datetime
import os
import sys
from loan_prediction.constants import *

from dataclasses import dataclass

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionCofig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR_NAME, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR_NAME, TRAIN_FILE)
    testing_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR_NAME, TEST_FILE)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_DIR,
                                               DATA_VALIDATION__REPORT_FILE_NAME)    
    


@dataclass
class DataTransformationConfig:
    data_tansformation_dir : str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path : str = os.path.join(data_tansformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME,
                                                     TRAIN_FILE.replace("csv","npy"))
    transformed_test_file_path : str = os.path.join(data_tansformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME,
                                                    TEST_FILE.replace("csv", "npy"))
    transformed_object_file_path : str = os.path.join(data_tansformation_dir, DATA_TRANSFORMATION_OBJECT_DIR_NAME,
                                                      PREPROSSER_OBJ_FILE_NAME)    
    



@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model = os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_NAME_FILE)