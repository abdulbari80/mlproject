import os
import sys
import pandas as pd
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTraining
from src.exception import CustomException   
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """A decorator class used to configure and manage data ingestion settings"""
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    """A class used to ingest, split and store data locally"""
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, test_size: float = 0.2):
        """This function ingests data a locally and splits it into train and test sets"""
        self.test_size = test_size
        logging.info("Data Ingestion method starts")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df = pd.read_csv('data\student.csv')
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=self.test_size, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Ingestion is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    print("This module is used for data ingestion",
           "and is not meant to be run directly")
