import os 
import sys

from dataclasses import dataclass

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path:str=os.path.join('artifacts','model.pk1')

class ModelTrainer:
    def  __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train,test):
        try:
            logging.info('model trainer initiated')
            X_train,X_test,y_train,y_test=(
                train[:,:-1],
                test[:,:-1],
                train[:,-1],
                test[:,-1]
            )
            logging.info(f'{X_train.shape},{X_test.shape},{y_train.shape},{y_test.shape}')

            models={
                'LGBMClassifier':LGBMClassifier(n_jobs=-1),
                'XGBClassifier':XGBClassifier(n_jobs=-1),
                'CatBoostClassifier':CatBoostClassifier(verbose=False)
            }

            params = {
            "LGBMClassifier": {
                'objective': ['binary', 'multiclass'],  # Specify the objective
                'learning_rate': [.1, .01, .05, .001],
                'num_leaves': [31, 63, 127],  # Number of leaves in one tree
                'max_depth': [-1, 5, 10],  # Limits the depth of the tree
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'bagging_fraction': [0.6, 0.7, 0.8, 0.9],  # Fraction of data for bagging
                'feature_fraction': [0.6, 0.7, 0.8, 0.9]  # Fraction of features for each iteration
            },
            "XGBClassifier": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'max_depth': [3, 5, 7],  # Maximum depth of a tree
                'subsample': [0.6, 0.7, 0.8, 0.9],  # Fraction of samples used for training each tree
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Fraction of features used for training each tree
            },
            "CatBoostClassifier": {
                'depth': [6, 8, 10],  # Depth of the tree
                'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
                'iterations': [30, 50, 100],  # Number of boosting iterations
                'l2_leaf_reg': [1, 3, 5],  # L2 regularization
                'bagging_temperature': [0.0, 0.1, 0.2, 0.5]  # Control over-fitting
            }
            
            }
            model_report:dict=evaluate_models(
                X_train=X_train,X_test=X_test,y_train=y_train,
                y_test=y_test,models=models,param=params
            )

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)




