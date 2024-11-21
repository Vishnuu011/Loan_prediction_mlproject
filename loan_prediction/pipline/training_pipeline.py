import sys 
import os
from loan_prediction.exception import CustomException
from loan_prediction.logger import logging

from loan_prediction.constants import *
from loan_prediction.entity.artifact_entity import DataIngestionArtifact
from loan_prediction.entity.config_entity import DataIngestionCofig

from loan_prediction.components.data_ingestion import DataIngestion


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionCofig()

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


    def run_pipeline(self, ) -> None:
        
       try:
          data_ingestion_artifact = self.start_data_ingestion()

       except Exception as e:
           raise CustomException(e, sys)
                  