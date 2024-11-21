import pymongo
import os
from loan_prediction.constants import DATA_BASE_NAME, CONNECTION_URL
from loan_prediction.logger import logging
from loan_prediction.exception import CustomException
import certifi
import sys

ca = certifi.where()

class MongoDBClient:
   
    client = None

    def __init__(self, database_name=DATA_BASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = CONNECTION_URL
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {CONNECTION_URL} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info("MongoDB connection succesfull")
        except Exception as e:
            raise CustomException(e,sys)
