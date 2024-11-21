from pandas import DataFrame
import numpy as np
import pandas as pd
from loan_prediction.constants import *
from loan_prediction.entity.config_entity import DataIngestionCofig

from loan_prediction.entity.artifact_entity import DataIngestionArtifact

from loan_prediction.data_access.loan_data import LoanData
from loan_prediction.exception import CustomException
from loan_prediction.logger import logging
from sklearn.model_selection import train_test_split
import sys
import os

class DataIngestion:

    def __init__(self, dataingestion_config : DataIngestionCofig=DataIngestionCofig()):
        try:
           self.data_ingetion_config = dataingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        

    def export_data_in_featurestore(self)->pd.DataFrame:
        try:
            logging.info(f"export data from mongodb")

            loan_data = LoanData()
            dataframe = loan_data.export_collection_as_dataframe(collection_name=self.data_ingetion_config.collection_name)
            
            logging.info(f"check data shape : {dataframe.shape}")

            feature_store_file_data = self.data_ingetion_config.feature_store_file_path
            dir_name = os.path.dirname(feature_store_file_data)
            os.makedirs(dir_name, exist_ok=True)

            logging.info(f"saving data into feature store {feature_store_file_data}")
            dataframe.to_csv(feature_store_file_data,index=False,header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e,sys)  


    def split_data_train_test(self, dataframe:pd.DataFrame)->None:

        try:
            logging.info("Entered train test split operation")

            train, test = train_test_split(dataframe, 
                   test_size=self.data_ingetion_config.train_test_split_ratio)
            
            logging.info("train test split operation complited .....")

            logging.info(f"checking train and test datashap : {train.shape}, {test.shape}")

            dir_name_t = os.path.dirname(self.data_ingetion_config.testing_file_path)
            os.makedirs(dir_name_t, exist_ok=True)
            train.to_csv(self.data_ingetion_config.training_file_path,index=False,header=True)
            test.to_csv(self.data_ingetion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train data and test data to path")
        except Exception as e:
            raise CustomException(e, sys) 


    def initiate_data_ingestion(self)-> DataIngestionArtifact:

        try:
            logging.info(f"Entered initiate_data_ingestion operation in data ingestion class")

            dataframe = self.export_data_in_featurestore()
            
            logging.info(f"got data in database ......")

            self.split_data_train_test(dataframe)

            logging.info(f"train and test data split perfomed")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path= self.data_ingetion_config.training_file_path,
                test_file_path= self.data_ingetion_config.testing_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)            


