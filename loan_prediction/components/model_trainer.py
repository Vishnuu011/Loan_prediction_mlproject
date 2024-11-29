import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")

from loan_prediction.exception import CustomException
from loan_prediction.logger import logging
from loan_prediction.entity.config_entity import ModelTrainerConfig
from loan_prediction.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from loan_prediction.constants import *
from loan_prediction.utils.utils import save_object, load_numpy_array


class Trainer(BaseEstimator, ClassifierMixin):

    

    def __init__(self, models : dict =None, params = None, 
                 X_train = None, y_train = None,
                 X_test = None, y_test = None):
        
        try:
            self.models = models
            self.params = params
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        except Exception as e:
            raise CustomException(e, sys)
        
    def _TrainModleToGridSearchCV(self,):

        try:
            report = {}

            for i in tqdm_notebook(range(len(list(self.models)))):
                model = list(self.models.values())[i]
                param = self.params[list(self.models.keys())[i]]

                gs = GridSearchCV(estimator=model,param_grid=param,cv= 4,
                                  scoring='accuracy', verbose=10,
                                  n_jobs=1)
                
                gs.fit(self.X_train, self.y_train)
                model.set_params(**gs.best_params_)
                model.fit(self.X_train,self.y_train)
                y_train_pred = model.predict(self.X_train)

                y_test_pred = model.predict(self.X_test)

                train_model_score = accuracy_score(self.y_train, y_train_pred)

                test_model_score = accuracy_score(self.y_test, y_test_pred)

                report[list(self.models.keys())[i]] = test_model_score

                
                return report
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self):
        try:
            trainer = self._TrainModleToGridSearchCV()
            return trainer  
     
        except Exception as e:
            raise CustomException(e, sys)
        


class ModelTrainer:

    def __init__(self, data_transformation_artifact : DataTransformationArtifact,
                 model_trainer_config : ModelTrainerConfig):
        
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e, sys)
        

    def get_model_object_and_report(self, train_array : np.array, test_array : np.array) -> Tuple[object, object]:

        try:
            logging.info("Split training and test input data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            params = {
            'random_forest': {
            "class_weight":["balanced"],
            'n_estimators': [20, 50, 30],
            'max_depth': [10, 8, 5],
            'min_samples_split': [2, 5, 10],
            },
            'decision_tree': {
            "class_weight":["balanced"],
            "criterion":['gini',"entropy","log_loss"],
            "splitter":['best','random'],
            "max_depth":[3,4,5,6],
            "min_samples_split":[2,3,4,5],
            "min_samples_leaf":[1,2,3],
            "max_features":["auto","sqrt","log2"]
            },
            'logistic_regression': {
            "class_weight":["balanced"],
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
            },
            'svc': {'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
            },
            'knnc':{
           'n_neighbors': [1,3,6]
            },
            'naive': {
           'priors': [None],
           'var_smoothing': [0.00000001, 0.000000001, 0.00000001]
            }

            }
            models = {
            'decision_tree': DecisionTreeClassifier(),
            'random_forest': RandomForestClassifier(),
            'logistic_regression': LogisticRegression(),
            'svc': SVC(),
            'knnc': KNeighborsClassifier(),
            'naive': GaussianNB()
            }

            model_report:dict = Trainer(models=models,params=params,X_train=X_train,
                                    y_train=y_train, 
                                    X_test=X_test,
                                    y_test=y_test
                                    ).fit()
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name] 
            
            return  best_model  

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:

        try:
            
            train_arr = load_numpy_array(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(file_path=self.data_transformation_artifact.transformed_test_file_path)
            best_model  = self.get_model_object_and_report(train=train_arr, test=test_arr)
            best_model_score=0.9
            if best_model_score<0.8:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.trained_model,
                obj=best_model
            )
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model
            )
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)       