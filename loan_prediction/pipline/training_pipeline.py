import sys 
import os
from loan_prediction.exception import CustomException
from loan_prediction.logger import logging

from loan_prediction.constants import *
from loan_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from loan_prediction.entity.config_entity import DataIngestionCofig, DataValidationConfig

from loan_prediction.components.data_ingestion import DataIngestion 
from loan_prediction.components.data_validation import DataValidation


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionCofig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(dataingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e 
        

    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, 
                                             data_validation_config=self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e 



    def run_pipeline(self, ) -> None:
        
       try:
          data_ingestion_artifact = self.start_data_ingestion()
          data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)

       except Exception as e:
           raise CustomException(e, sys)
                  