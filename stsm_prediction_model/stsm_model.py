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
from stsm_prediction_model.error_handling import ModelError
from stsm_prediction_model.error_handling import DBError
from stsm_prediction_model.error_handling import UserRequestError

from logger import Logger
#C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\


class StsmPredictionModel:
    def __init__(self,logger):
        self.model = None
        self.db_conn = None
        self.logger = logger

    def load_model(self):
        try:
            self.model =joblib.load('C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\\trained_model2.pkl')
            self.logger.info('model loaded successfully')
            return
        except:
            raise ModelError('Failed loading saved model', 1000, str(sys.exc_info()[1]))

    def connect(self):
        try:
            self.db_conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                                           , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
            self.logger.info('Connected to DB')
        except:
            raise DBError('DB connection failed', 1001, str(sys.exc_info()[1]))

    def disconnect(self):
        if self.db_conn:
            try:
                self.db_conn.close()
                self.logger.info('Disconnected from DB')
            except:
                raise DBError('DB disconnection failed ', 1002, str(sys.exc_info()[1]))

    def predict(self, request):
        self.logger.info("In Stsm Model evaluate:")
        try:
            request_signals = prediction_request_signals(request, self.db_conn)
            self.logger.info('Finished loading request eeg signals. size = %s' % str(np.shape(request_signals)))
        except (DBError, UserRequestError):
            raise
        try:
            self.logger.info('Starting prediction...')
            prediction = self.model.predict(request_signals)
            self.logger.info('Finished prediction')
        except:
            raise ModelError('Error in model.predict', 1005, sys.exc_info()[1])
        try:
            predictions_db(prediction,request,self.db_conn)
            self.logger.info('Predictions successfully uploaded to DB')
        except (DBError, UserRequestError):
            raise
        return

    def validate(self, request):
        self.logger.info("in Stsm Model validate:")
        try:
            validation_file_name = validate_user_results(request, self.db_conn)
            self.logger.info('Finished validation - results file created')
        except (ModelError, DBError, UserRequestError):
            raise
        return validation_file_name
