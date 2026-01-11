import os
import sys
from src.pipeline.execption import customException
from src.pipeline.logger import logger
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.pipeline.utils import save_obj

@dataclass
class DataTransformationconfig():
    preprocessor_obj_path= os.path.join('artifacts','preprocessor.pkl')

class DataTransform():
    def __init__(self):
        self.DataTransformation_config=DataTransformationconfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )
            logger.info("numerical trnsformation done:)")
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ("onehot",OneHotEncoder(handle_unknown='ignore')),
                    ("scaling",StandardScaler(with_mean=False))
                ]
            )
            logger.info("categorical transformation complete")

            preprocessor=ColumnTransformer(
             [
                ("num_pipeline",numerical_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
             ]
             )
            return preprocessor

        except Exception as e:
            raise customException(e,sys)
        
    def init_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            # normalize column names to snake_case to make downstream column references consistent
            train_df.columns = [col.strip().lower().replace(' ', '_').replace('/', '_') for col in train_df.columns]
            test_df.columns = [col.strip().lower().replace(' ', '_').replace('/', '_') for col in test_df.columns]

            logger.info("read train dataset")
            logger.info("read test dataser")

            preprocesser=self.get_data_transformer()
            target_columns='math_score'
            numerical_columns=['writing_score','reading_score']
            input_feature_tr=train_df.drop(columns=[target_columns],axis=1)
            target_columns_tr=train_df[target_columns]

            input_feature_test=test_df.drop(columns=[target_columns],axis=1)
            target_columns_test=test_df[target_columns]

            input_transformed_tr=preprocesser.fit_transform(input_feature_tr)
            input_transformed_test=preprocesser.transform(input_feature_test)

            train_arr=np.c_[
                input_transformed_tr,np.array(target_columns_tr)
            ]
            test_arr=np.c_[
                input_transformed_test,np.array(target_columns_test)
            ]
            logger.info("preprocessing done")

            save_obj(
                    file_path=self.DataTransformation_config.preprocessor_obj_path,
                    obj=preprocesser
            )
            return train_arr, test_arr, self.DataTransformation_config.preprocessor_obj_path

        except Exception as e:
            raise customException(e,sys)
            


        