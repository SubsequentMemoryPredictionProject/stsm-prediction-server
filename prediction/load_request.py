from sklearn.externals import joblib
import os
import sys
import pymysql.connections


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from DB.db_access import get_signals
import config as cfg


conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                       , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

def load_model():
    mlp_model = joblib.load('mlp_model.pkl')
    return mlp_model


def prediction_request(request,conn):
    request_signals =[]
    print(len(request))
    for i in range(len(request)):
        print(request[i][0])
        print(request[i][1])
        print(request[i][2])
        prediction_details = " AND user_id=" + str(request[i][0]) + " AND subject_id=" + str(request[i][1])\
                         + " AND word_id=" + str(request[i][2])
        #TODO change table to user_data
        request_signals.append(get_signals(conn,prediction_details,'data_set'))
    return request_signals
