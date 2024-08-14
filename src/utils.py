import os
import sys

import pandas as pd
import numpy as np 

import dill 
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV


from src.exception import  CustomException
from src.logger import logging 

def save_object(file_path,obj):
    try:
        dir_path =os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train,y_train,X_test,y_test,model,param):
    try:
        
        rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param,
                n_iter=100,  # Number of parameter settings sampled
                scoring='accuracy',  # Metric to optimize
                cv=3,  
                verbose=1,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
        
        rs.fit(X_train,y_train)
        model.set_params(**rs.best_params_)
        model.fit(X_train,y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)
        report= test_model_score

        return report,model

    except Exception as e:
        raise CustomException(e, sys)

def load_data(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)