from sklearn.externals import joblib
import sys
import os
import pymysql
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from prediction.load_request import prediction_request_signals
from prediction.report_predictions import predictions_db
import config as cfg
from model_evaluation.validation_report import validate_user_results
from logger import Logger
#C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\


class StsmPredictionModel:
    def __init__(self,logger):
        self.model = None
        self.db_conn = None
        self.logger = logger

    def load_model(self):
        try:
            self.model = joblib.load('C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\mlp_model.pkl')
            self.logger.info('Model loaded successfully')
        except:
            self.logger.error('Error loading model - %s' % str(sys.exc_info()))

    def connect(self):
        try:
            self.db_conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                             , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
            self.logger.info('Connected to DB')
        except:
            self.logger.error('Error connecting to DB - %s' % str(sys.exc_info()))

    def disconnect(self):
        self.db_conn.close()
        self.logger.info('Disconnected from DB')

    def evaluate(self, request):
        try:
            self.logger.info("In Stsm Model evaluate:")
            request_signals = prediction_request_signals(request, self.db_conn)
            self.logger.info('Finished loading request eeg signals. size = %s'% str(np.shape(request_signals)))
            self.logger.info('Starting prediction...')
            prediction = self.model.predict(request_signals)
            self.logger.info('Finished prediction')
            predictions_db(prediction,request,self.db_conn)
            self.logger.info('Finished updating predictions to DB')
            return
        except:
            self.logger.error('Error in prediction - %s' %sys.exc_info())
            return

    def validate(self,request):
        self.logger.info("in Stsm Model validate:")
        try:
            validation_file = validate_user_results(request,self.db_conn)
            self.logger.info('Finished validation - results file created')
            return validation_file
        except:
            self.logger.error('Error in validation - %s' %str(sys.exc_info()))




