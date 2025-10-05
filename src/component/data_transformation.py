import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transormation_config = DataTransformationConfig()

    def get_transformer_obj(self):
        logging.info('Data preprocessing starts')
        try:
            num_columns = ["writing_score", "reading_score"]
            cat_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline = Pipeline(
                steps = [
                    # Median is chosen because it is robust to outliers
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('cat_pipeline', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_columns),
                    ('cat_pipeline', cat_pipeline, cat_columns)
                ]
            )
            logging.info('Data preprocessing pipeline created.')
            return preprocessor           
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """ This saves fitted proprocessing model
        Parameter:
            train_data_path(str): training data file path
            test_data_path(str): testing data file path

        Return:
            (array): train_arr: transformed training array
            (array): test_arr: transformed testing array
            (str): preprocessor_obj_path: saved preprocessor object file path
        """
        try:
            df_train = pd.read_csv(train_data_path)
            df_test = pd.read_csv(test_data_path)
            target_variable = 'math_score'
            X_train = df_train.drop(columns=[target_variable], axis=1)
            y_train = df_train[target_variable]
            X_test = df_test.drop(columns=[target_variable], axis=1)
            y_test = df_test[target_variable]

            preprocess_obj = self.get_transformer_obj()
            X_train_arr = preprocess_obj.fit_transform(X_train)
            X_test_arr = preprocess_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            preprocess_obj_path = self.data_transormation_config.preprocessor_obj_file_path

            save_object(file_path=preprocess_obj_path, obj=preprocess_obj)
            logging.info("Preprocessing object saved.")
            return(
                train_arr,
                test_arr,
                preprocess_obj_path
                )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    print("This module provides data transformation"
           " and is not meant to be run directly.")
    