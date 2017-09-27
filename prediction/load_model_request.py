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


def prediction_request(request,db):
    prediction_details = "AND user_id=" + str(request[0]) + " AND subject_id=" + str(request[1])\
                         + " AND word_id=" + str(request[2])
    user_request = get_signals(db,prediction_details)
    return user_request

req = [1, 2, 50]
model = load_model()
features_to_predict = prediction_request(conn,req)
for i in features_to_predict:
    print(i)
prediction = model.predict(features_to_predict)
print(prediction)