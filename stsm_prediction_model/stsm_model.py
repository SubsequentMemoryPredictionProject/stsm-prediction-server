from sklearn.externals import joblib
import sys
import os
import pymysql
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from prediction.load_request import prediction_request
from prediction.report_predictions import results_db
import config as cfg
from model_evaluation.validation import validate_user_results
from logger import Logger


class StsmPredictionModel:
    def __init__(self):
        self.model = None
        self.db_conn = None
        self.logger = Logger().get_logger()

    def load_model(self):
        try:
            self.model = joblib.load('C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\mlp_model.pkl')
            self.logger.info('Model loaded successfully')
        except:
            self.logger.error('Error loading model - %s' % sys.exc_info()[0])

    def connect(self):
        try:
            self.db_conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                             , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
            self.logger.info('Connected to DB')
        except:
            self.logger.error('Error connecting to DB - %s' % sys.exc_info()[0])

    def disconnect(self):
        self.db_conn.close()
        self.logger.info('Disconnected from DB')

    def evaluate(self, request):
        try:
            self.logger.info("Stsm Model evaluate:")
            request_signals = prediction_request(request, self.db_conn)
            self.logger.info('Finished loading request eeg signals. size = %s'% str(np.shape(request_signals)))
            self.logger.info('Starting prediction...')
            prediction = self.model.predict(request_signals)
            self.logger.info('Finished prediction')
            results_db(prediction,request,self.db_conn)
            return prediction
        except:
            self.disconnect()
            print(sys.exc_info())



    def validate(self,results):
        validate_user_results(results,self.db_conn)
        return




