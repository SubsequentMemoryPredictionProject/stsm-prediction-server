from sklearn.externals import joblib
import os
import sys
import ast
import pymysql.connections
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from DB.db_access import get_signals
from model_evaluation.validation_report import create_user_query


def prediction_request_signals(request,conn):
    prediction_details = create_user_query(request)
    #TODO change table to user_data
    request_signals = get_signals(conn,prediction_details,'data_set')
    return request_signals
