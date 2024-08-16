import os
import sys
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
from src.exception import CustomException
from src.logger import logging


try:
    logging.info('testing of predict_pipeline has started')
    data_obj=CustomData('SLAUSON                      AV','HALLDALE                     AV',33.9892,-118.3024,
    '01/24/2020 12:00:00 AM','01/24/2020 12:00:00 AM',2300,12,'77th Street',1235,2,'1213 0400 1822',0,'X','X',
    102,'SIDEWALK',400.0,'STRONG-ARM (HANDS, FIST, FEET OR BODILY FORCE)','IC','Invest Cont')
    data=data_obj.get_feature_as_dataframe()
    
    model_obj=PredictPipeline()

    result=model_obj.predict(features=data)

    logging.info(f'testing is done')

except  Exception as e:
    raise CustomException(e,sys)
