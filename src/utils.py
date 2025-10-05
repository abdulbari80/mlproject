import os
import sys 
import pickle
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.logger import logging
# import numpy as np
# import pandas as pd

def evaluate_models(models, params, X_train, X_test, y_train, y_test):
    """This function fintunes parameters and evaluates performance among the model"""
    try:
        performance_report = {}
        for key, value in models.items():
            model = value
            param = params[key]
            gsc = GridSearchCV(model, param, cv=3)
            gsc.fit(X_train, y_train)
            model.set_params(**gsc.best_params_)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            performance_report[key] = r2_score(y_test, y_test_pred)
            logging.info(f"{key} has been evaluated!")
        return performance_report
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    """This function saves object in the given path"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    """This function loads object from the given path"""
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    print("This contains utility functions only",
           "and is not meant to be run directly")