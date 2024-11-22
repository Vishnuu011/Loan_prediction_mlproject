import os
import sys
import pandas as pd
import numpy as np
import json
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from pandas import DataFrame
from loan_prediction.constants import SCHEMA_FILE_PATH
from loan_prediction.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from loan_prediction.entity.config_entity import DataValidationConfig
from loan_prediction.exception import CustomException
from loan_prediction.logger import logging
from loan_prediction.utils.utils import read_yml_file, write_yaml_file


class DataValidation:

    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_arifact = data_ingestion_artifact
            self.data_ingestion_config = data_validation_config
            self._schema_config = read_yml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)
        
    def validate_number_columns(self, dataframe:pd.DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"validation status is : {status}")
            return status
        except Exception as e:
            raise CustomException(e, sys) 

    def is_columns_exist(self, df:pd.DataFrame) -> bool:
        try:
            dataframe_col = df.columns
            missing_numerical_col = []
            missing_categorical_col = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_col:
                    missing_numerical_col.append(column)

                if len(missing_numerical_col) > 0:
                    logging.info(f"Missing numeric colums : {missing_numerical_col}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_col:
                    missing_categorical_col.append(column)

                if len(missing_categorical_col) > 0:
                    logging.info(f"Missing categorical colums : {missing_categorical_col}")

            return False if len(missing_numerical_col) > 0 or len(missing_categorical_col) > 0 else True                  

        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame, ) -> bool:
        
        try:
            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(reference_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=self.data_ingestion_config.drift_report_file_path, content=json_report)

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:

        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_arifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_arifact.test_file_path))

            status = self.validate_number_columns(dataframe=train_df)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.validate_number_columns(dataframe=test_df)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.is_columns_exist(df=train_df)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.is_columns_exist(df=test_df)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_dataset_drift(train_df, test_df)
                if drift_status:
                    logging.info(f"Drift detected.")
                    validation_error_msg = "Drift detected"
                else:
                    validation_error_msg = "Drift not detected"
            else:
                logging.info(f"Validation_error: {validation_error_msg}")
                

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                drift_report_file_path=self.data_ingestion_config.drift_report_file_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e       
                   