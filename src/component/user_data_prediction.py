import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class ProcessUserData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parent_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_frame(self):
        try:
            user_input_dict = {
                'gender': self.gender,
                'race_ethnicity':self.race_ethnicity,
                'parental_level_of_education': self.parent_education,
                'lunch': self.lunch,
                'test_preparation_course': self.test_preparation,
                'reading_score': self.reading_score,
                'writing_score': self.writing_score
            }
            df_user_input = pd.DataFrame(user_input_dict, index=[0])
            return df_user_input
        except Exception as e:
            raise CustomException(e, sys)

class Prediction:
    def __init__(self):
        pass

    def get_prediction(self, df_user_input):
        try:
            # preprocess user inputs
            data_preprocessing_path = os.path.join('artifacts', 'preprocessor.pkl')
            preprocessor = load_object(data_preprocessing_path)
            preproc_user_data = preprocessor.transform(df_user_input)

            # extract model prodiction
            model = load_object(os.path.join('artifacts', 'model.pkl'))
            pred_result = model.predict(preproc_user_data)
            logging.info("Model prediction successful")
            return pred_result

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    print("This module processes user input data",
           "and makes prediction using trained model",
           "and is not meant to be run directly")