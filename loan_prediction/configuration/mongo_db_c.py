import pymongo
import os
from loan_prediction.constants import DATA_BASE_NAME, CONNECTION_URL
from loan_prediction.logger import logging
from loan_prediction.exception import CustomException
import certifi

ca = certifi.where()

class MongoDBClient:
    
    client = None
    def __init__(self, database_name=DATA_BASE_NAME):
        try:
          if MongoDBClient.client is None:
            mongo_db_url = os.path.join(CONNECTION_URL,database_name)
            if mongo_db_url is not None:
                raise Exception("Mongo db connection is not established")
            MongoDBClient.client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info(f"Connection established with: {database_name}")
        except Exception as e:
            raise CustomException
