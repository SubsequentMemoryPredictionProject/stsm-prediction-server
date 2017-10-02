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


class StsmPredictionModel:
    def __init__(self):
        self.model = None
        self.db_conn = None

    def load_model(self):
        try:
            self.model = joblib.load(
            'C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\mlp_model.pkl')
        except:
            print(sys.exc_info()[0])

    def connect(self):
        self.db_conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                             , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    def disconnect(self):
        self.db_conn.close()

    def evaluate(self, request):
        try:
            self.connect()
            request_signals = prediction_request(request, self.db_conn)
            print('finished request')
            print(len(request_signals))
            print(np.shape(request_signals))
            prediction = self.model.predict(request_signals)
            print('got predictions')
            print(prediction)
            results_db(prediction,request,self.db_conn)
            self.disconnect()
            return prediction
        except:
            print(sys.exc_info()[0])
            self.disconnect()

    def validate(self,results):
        self.connect()
        validate_user_results(results,self.db_conn)
        self.disconnect()
        return




