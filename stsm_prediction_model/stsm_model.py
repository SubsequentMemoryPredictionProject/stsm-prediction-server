from sklearn.externals import joblib
import sys
import os
import pymysql


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from prediction.load_request import prediction_request
from prediction.report_predictions import results_db
import config as cfg
from model_evaluation.validation import validate_user_results


class StsmPredictionModel:
    def __init__(self):
        self.model = joblib.load('C:\\Users\\user\PycharmProjects\stsm-prediction-server\learning\mlp_model\mlp_model.pkl')
        self.db_conn = None

    def connect(self):
        self.db_conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                             , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    def disconnect(self):
        self.db_conn.close()

    def evaluate(self,request):
        print(request)
        self.connect()
        user_request = prediction_request(request,self.db_conn)
        prediction = self.model.predict(user_request)
        print('got predictions')
        results_db(prediction,request,self.db_conn)
        self.disconnect()
        return prediction

    def validate(self,results):
        self.connect()
        validate_user_results(results,self.db_conn)
        self.disconnect()
        return




