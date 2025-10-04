import os
import sys
from dataclasses import dataclass   

from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor,
                              AdaBoostRegressor,
                              GradientBoostingRegressor)
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression   
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTraining:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()

    def model_trainer(self, train_arr, test_arr, r2_threshold=0.6):
        """This method finds  and saves the best model"""
        self.r2_threshold = r2_threshold
        try:
            X_train_arr = train_arr[:,:-1]
            y_train_arr = train_arr[:,-1]
            X_test_arr = test_arr[:,:-1]
            y_test_arr = test_arr[:,-1]
            logging.info('Instantiate models and set hyperparameters')
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoosting': XGBRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'CatBoost': CatBoostRegressor(verbose=False)
            }

            params = {
                'Linear Regression': {},
                'Decision Tree': {},
                'Random Forest': {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'Gradient Boosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'subsample': [0.6, 0.7, 0.8, 0.9]
                },
                'XGBoosting': {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'AdaBoost': {
                    'learning_rate': [0.1, 0.05,0.01, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                'CatBoost': {
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'depth': [6, 8, 10],
                    'iterations': [30, 50, 100]
                }
            }
            logging.info("Model evaluation starts.....")
            results = evaluate_models(models=models, params=params, 
                                      X_train=X_test_arr, y_train=y_test_arr, 
                                      X_test=X_test_arr, y_test=y_test_arr)
            logging.info("Model evaluation finished!")
            best_model = max(results, key=results.get)
            best_model_r2 = results[best_model]
            if best_model_r2 < self.r2_threshold:
                logging.info("Waring!! Test R2 is less than {self.r2_threshold}")
            best_model_path = self.model_train_config.model_obj_file_path
            save_obj(best_model_path, best_model)
            return (best_model, best_model_r2)
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    print("This trains ML models and is not meant to be run directly.")