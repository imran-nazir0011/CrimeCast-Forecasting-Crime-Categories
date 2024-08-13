import os 
import sys

import numpy as np 

from dataclasses import dataclass

from lightgbm import LGBMClassifier

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

    def initiate_model_trainer(self,train,test,train_label,test_label):
        try:
            logging.info('model trainer initiated')
            X_train,X_test,y_train,y_test=(train,test,train_label,test_label)

            model=LGBMClassifier(n_jobs=-1)

            params = {
                'num_leaves': np.arange(20, 150, 10),
                'learning_rate': np.logspace(-3, 0, 100),
                'n_estimators': np.arange(50, 1000, 50),
                'min_child_samples': np.arange(5, 50, 5)
            }           


            best_model_score,best_model=evaluate_models(
                X_train=X_train,X_test=X_test,y_train=y_train.ravel(),
                y_test=y_test.ravel(),model=model,param=params
            )


            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        except Exception as e:
            raise CustomException(e,sys)




