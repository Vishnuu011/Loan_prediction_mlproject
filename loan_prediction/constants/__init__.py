from datetime import datetime
import os

DATA_BASE_NAME="LOAN_DATA"
COLLECTION_NAME="loan_data"
CONNECTION_URL="mongodb+srv://Vishnurrajeev:<password>@cluster0.hsmya.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

PIPELINE_NAME : str = "loanpredictionproject"
ARTIFACT_DIR : str = "artifact"

MODEL_NAME_FILE = "model.pkl"
FILE_NAME = "loan_data.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv" 
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml")
PREPROSSER_OBJ_FILE_NAME = "preprocesser.pkl"
TRAGET_COL = "loan_status"

#DATAINGESTION RELATED CONSTANTS

DATA_INGESTION_COLLECTION_NAME: str ="loan_data"
DATA_INGESTION_DIR_NAME: str ="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str ="feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str ="ingested"
DATA_INGESTION_TRAIN_DIR_NAME: str ="train"
DATA_INGESTION_TEST_DIR_NAME: str ="test"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float =0.2
DATA_INGESTION_DATA_BASE_NAME = "LOAN_DATA"


# DATA VALIDATION RELATED CONSTANTS


DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_DIR: str = "drift_report"
DATA_VALIDATION__REPORT_FILE_NAME: str = "report.yaml"

# DATA TRANSFORMATION RELATED CONSTANTS

DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME : str = "transform"
DATA_TRANSFORMATION_OBJECT_DIR_NAME : str = "transformed_obj"

# MODEL TRAINER RELATED CONSTANTS

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
