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
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
                   