import os
import sys
import sklearn
import pandas as pd
import numpy as np
from loan_prediction.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from loan_prediction.entity.config_entity import ModelTrainerConfig, DataTransformationConfig
from loan_prediction.constants import *

from loan_prediction.exception import CustomException

from loan_prediction.utils.utils import load_object



class predictPipline:

    def __init__(self,):

        self.model = ModelTrainerConfig()
        self.datatransformation = DataTransformationConfig()
        
    def predict(self, feature):

        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocesser.pkl')
            model = load_object(file_path=model_path)
            preprosser = load_object(file_path=preprocessor_path)

            scaled_data = preprosser.transform(feature)
            preds = model.predict(scaled_data)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(  self,
        person_age: int,
        person_gender: str,
        person_education:str,
        person_income: float,
        person_emp_exp: int,
        person_home_ownership: str,
        loan_amnt: int,
        loan_intent: str,
        loan_int_rate : float,
        loan_percent_income :float,
        cb_person_cred_hist_length:float,
        credit_score:int,
        previous_loan_defaults_on_file:str):

        self.person_age = person_age

        self.person_gender = person_gender

        self.person_education = person_education

        self.person_income = person_income

        self.person_emp_exp = person_emp_exp

        self.person_home_ownership = person_home_ownership

        self.loan_amnt = loan_amnt

        self.loan_intent =loan_intent

        self.loan_int_rate = loan_int_rate

        self.loan_percent_income = loan_percent_income

        self.cb_person_cred_hist_length = cb_person_cred_hist_length

        self.credit_score = credit_score

        self.previous_loan_defaults_on_file = previous_loan_defaults_on_file

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "person_age": [self.person_age],
                "person_gender": [self.person_gender],
                "person_education": [self.person_education],
                "person_income": [self.person_income],
                "person_emp_exp": [self.person_emp_exp],
                "person_home_ownership": [self.person_home_ownership],
                "loan_amnt": [self.loan_amnt],
                "loan_intent" : [self.loan_intent],
                "loan_int_rate" : [self.loan_int_rate],
                "loan_percent_income" : [self.loan_percent_income],
                "cb_person_cred_hist_length" : [self.cb_person_cred_hist_length],
                "credit_score" : [self.credit_score],
                "previous_loan_defaults_on_file" : [self.previous_loan_defaults_on_file]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


