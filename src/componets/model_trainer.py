import os
import sys

from catboost import CatBoostRegressor
from src.pipeline.execption import customException
from src.pipeline.logger import logger
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model  import Ridge,Lasso,ElasticNet
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.pipeline.utils import save_obj
from src.pipeline.utils import evaluvate_models

@dataclass
class ModelTrainingconfig():
    train_model_path=os.path.join('artifacts','model.pkl')

class modelTrainer():
    def __init__(self):
        self.model_trainer_config=ModelTrainingconfig()

    def init_model_train(self,train_array,test_array):
        try:
            logger.info("model traing started")
            # split features and target correctly
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models={
                "linear_regression":LinearRegression(),
                "decision_tree":DecisionTreeRegressor(),
                "random_forest":RandomForestRegressor(),
                "gradient_boosting":GradientBoostingRegressor(),
                "k_neighbors":KNeighborsRegressor(),
                "xgboost":XGBRegressor(),
                "adaboost":AdaBoostRegressor(),
                "catboost":CatBoostRegressor()

            }
            # keep things simple: no hyperparameter search; use default parameters

            # call evaluvate_models which returns simple scalar R2 scores
            model_report: dict = evaluvate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models)

            # pick the best model by R2 score (simple and readable)
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise customException("no best model found", sys)

            logger.info(f"best model found {best_model_name} with score {best_model_score}")

            save_obj(
                file_path=self.model_trainer_config.train_model_path,
                obj=best_model
            )

            # return a simple result dict
            return {
                "best_model_name": best_model_name,
                "best_model_score": float(best_model_score)
            }





        except Exception as e:
            raise customException(e,sys)

        