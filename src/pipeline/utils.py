import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from src.pipeline.execption import customException
import dill

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise customException(e, sys)

def evaluvate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            mae=mean_absolute_error(y_test, y_pred)
            mse=mean_squared_error(y_test, y_pred)
            rmase=np.sqrt(mse)
            # store only the R2 score for each model so outcomes are scalars (not tuples)
            report[model_name] = r2_square
        return report
    except Exception as e:
        raise customException(e, sys)