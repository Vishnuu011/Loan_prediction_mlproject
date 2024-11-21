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
         self.mongo_client = MongoDBClient(database_name=DATA_BASE_NAME)
      except Exception as e:
         raise CustomException(e,sys)
        

   def export_collection_as_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:
        try:
         if database_name is None:
            collection = self.mongo_client.database[collection_name]
         else:
            collection = self.mongo_client[database_name][collection_name]

         df = pd.DataFrame(list(collection.find()))
            #if "_id" in df.columns.to_list():
               #df = df.drop(columns=["_id"], axis=1)
            #df.replace({"na":np.nan},inplace=True)
         return df  
        except Exception as e:
            raise CustomException(e,sys)     