import os
import sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN

from loan_prediction.logger import logging
from loan_prediction.exception import CustomException
from loan_prediction.entity.config_entity import DataTransformationConfig
from loan_prediction.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from loan_prediction.constants import *
from loan_prediction.utils.utils import read_yml_file, save_numpy_array, save_object

class DataTransformation:

    def __init__(self, data_ingestion_artifact : DataIngestionArtifact, data_transformation_config : DataTransformationConfig,
                 data_validation_arifact : DataValidationArtifact):
        
       try:
           self.data_ingestion_artifact = data_ingestion_artifact
           self.data_transformation_config = data_transformation_config
           self.data_validation_arifact = data_validation_arifact
           self._schema_config = read_yml_file(file_path=SCHEMA_FILE_PATH)
       except Exception as e:
           raise CustomException(e, sys)
       
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) 
          
    def get_data_transform_object(self) -> Pipeline:

        logging.info( 
            "Entered data transformation object operation .........."
            )
        
        try:
            numeric_transformer = StandardScaler()
            ordinal_encoder = OrdinalEncoder()
            
            
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['transform_columns']
            num_features = self._schema_config['num_features']

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocesser = ColumnTransformer([
                ("Ordinal_Encoder", ordinal_encoder, or_columns),
                ("Transformer", transform_pipe, transform_columns),
                ("StandardScaler", numeric_transformer, num_features)
            ])
            logging.info("Created preprocessor object from ColumnTransformer")
            logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocesser

        except Exception as e:
            raise CustomException(e, sys)
        
    def remove_outliers_IOR(self, col, df):
        try:
           Q1 = df[col].quantile(0.25)
           Q3 = df[col].quantile(0.75)

           iqr = Q3 - Q1

           upper_limit = Q3 + 1.5 * iqr
           lower_limit = Q1 - 1.5 * iqr

           df.loc[(df[col]>upper_limit),col] = upper_limit
           df.loc[(df[col]<lower_limit),col] = lower_limit

           return df
        except Exception as e:
            raise CustomException(e, sys)    
        
    def initiate_data_transformation(self,) -> DataTransformationArtifact:

        try:
            logging.info("Enter initiate_data_transformation operation")

            num_features_out = self._schema_config['num_features']

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            preprocesser = self.get_data_transform_object()
            logging.info("Entered remove_outliers_IQR operation .....")

            for features in num_features_out:
                self.remove_outliers_IOR(col=features, df=train_df )

            logging.info("remove_outliers_IQR train_df performed sucessfuly...")

            for features in num_features_out:
                self.remove_outliers_IOR(col=features, df=test_df )

            logging.info("remove_outliers_IQR test_df performed sucessfuly...")

            input_features_train_df = train_df.drop(columns=[TRAGET_COL], axis=1)
            input_target_features_train_df = train_df[TRAGET_COL] 

            logging.info("Got train features and test features of Training dataset")

            input_features_test_df = test_df.drop(columns=[TRAGET_COL], axis=1)
            input_target_features_test_df = test_df[TRAGET_COL]  

            logging.info(
                  "Applying preprocessing object on training dataframe and testing dataframe"
                ) 
            input_train_arr = preprocesser.fit_transform(input_features_train_df)

            logging.info(
                  "Applying preprocessing object on training dataframe and testing dataframe"
                ) 
        
            input_test_arr = preprocesser.transform(input_features_test_df)

        
            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTEENN on Training dataset")

            smt = SMOTEENN(sampling_strategy="minority")

            input_features_train_final, target_feature_train_final = smt.fit_resample(
                input_train_arr, input_target_features_train_df
            )

            logging.info("Applied SMOTEENN on training dataset")

            logging.info("Applying SMOTEENN on testing dataset")

            input_features_test_final, target_feature_test_final = smt.fit_resample(
                input_test_arr, input_target_features_test_df
            )

        
            logging.info("Applied SMOTEENN on testing dataset")

            logging.info("Created train array and test array")

            train_arr = np.c_[
               input_features_train_final, np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
               input_features_test_final, np.array(target_feature_test_final)
            ]

            save_object(
                self.data_transformation_config.transformed_object_file_path, 
                preprocesser
            )

            save_numpy_array(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            logging.info("Saved the preprocessor object")

            logging.info(
                "Exited initiate_data_transformation method of Data_Transformation class"
            )
        
            data_transformation_arifact = DataTransformationArtifact(
                transformed_obj_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_arifact


        except Exception as e:
            raise CustomException(e, sys)



