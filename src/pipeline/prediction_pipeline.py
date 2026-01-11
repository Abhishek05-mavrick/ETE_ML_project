import os
import sys
from src.pipeline.execption import customException
from src.pipeline.logger import logger
from src.pipeline.utils import load_obj
import numpy as np
import pandas as pd


class predictpipeline():
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)

            # ensure features contain the same columns used during training
            expected_cols = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
                'reading_score',
                'writing_score'
            ]
            # if input is a DataFrame, add missing columns (fill with NaN) and reorder
            if isinstance(features, pd.DataFrame):
                for c in expected_cols:
                    if c not in features.columns:
                        features[c] = np.nan
                features = features[expected_cols]

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise customException(e, sys) 
    
class customData():
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
         try:
            custom_data = {
                    "gender": [self.gender],
                    "race_ethnicity": [self.race_ethnicity],
                    "parental_level_of_education": [self.parental_level_of_education],
                    "lunch": [self.lunch],
                    "test_preparation_course": [self.test_preparation_course],
                    "reading_score": [self.reading_score],
                    "writing_score": [self.writing_score]
                }
            return pd.DataFrame(custom_data)
         except Exception as e:
             raise customException(e,sys)
        

