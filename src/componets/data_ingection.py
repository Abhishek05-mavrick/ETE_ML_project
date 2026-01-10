from sklearn.model_selection import train_test_split
import os
import sys
from  src.pipeline.execption import customException
from  src.pipeline.logger import logger
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngection():
    def __init__(self):
        self.ingection_config=DataIngestionConfig()

    def initiate(self):
        logger.info("Data Ingection started")
        try:
            df=pd.read_csv('Notebook/data/StudentsPerformance.csv')
            logger.info("reading data")

            os.makedirs(os.path.dirname(self.ingection_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingection_config.raw_data_path,index=False,header=True)
            logger.info("Train test spilt initiated")
            train_Set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            test_set.to_csv(self.ingection_config.test_data_path,index=False,header=True)
            train_Set.to_csv(self.ingection_config.train_data_path,index=False,header=True)

            logger.info('THE ingestion sucessfull')

            return(
              self.ingection_config.train_data_path,
              self.ingection_config.test_data_path
            )
        except Exception as e:
            raise customException(e,sys)
        


if __name__=="__main__":
    obj=DataIngection()
    obj.initiate()
