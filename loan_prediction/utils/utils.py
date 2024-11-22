import yaml
from pandas import DataFrame
import pickle
import numpy as np
import sys
import os

from loan_prediction.logger import logging
from loan_prediction.exception import CustomException



def read_yml_file(file_path: str)-> dict:
    try:
       with open(file_path,'rb') as yaml_file:
        return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys) from e  


def load_object(file_path: str)-> object:
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) from e    

def save_numpy_array(file_path: str, array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise CustomException(e,sys) from e 

def load_numpy_array(file_path: str)-> np.array:
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) from e

def save_object(file_path: str, obj: object)-> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys) from e 

def drop_columns(df: DataFrame, cols:list)-> DataFrame:
    logging.info("Entered drop_columns methon of utils")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info("Entered drop_columns methon of utils")
        return df
    except Exception as e:
        raise CustomException(e,sys) from e

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys) from e     