import numpy as np
import pandas as pd
import sys
from loan_prediction.configuration.mongo_db_c import MongoDBClient
from loan_prediction.constants import *
from loan_prediction.logger import logging
from loan_prediction.exception import CustomException
from typing import Optional

class LoanData:

    def __init__(self):

        try:
           mongo_db_client = MongoDBClient(DATA_BASE_NAME)
        except Exception as e:
            raise CustomException(e,sys) 
    
    def export_collection_as_dataframe(self, collection_name:str, database_name:Optional[str])->pd.DataFrame:
        try:
           if database_name is None:
              collection = self.mongo_db_client.database[collection_name]
           else:
                collection = self.mongo_db_client[database_name][collection_name]

                df = pd.DataFrame(list(collection.find()))
           return df  
        except Exception as e:
            raise CustomException(e,sys)     
