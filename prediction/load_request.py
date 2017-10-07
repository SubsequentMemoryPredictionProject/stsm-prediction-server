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
from logger import Logger

logger = Logger().get_logger()


def prediction_request_signals(request,conn):
    msg = 'success'
    logger.info('In prediction_request_signals')
    logger.info("User request: %s" % request)
    if not request:
        msg = "user request is empty"
        return [], msg
    prediction_details = create_user_query(request)
    request_signals = get_signals(conn,prediction_details,'user_data')
    return request_signals


def check_request(request):
    if not request:
        msg = "user request is empty"
    user_id = request['user_id']

    if not user_id:
        msg = 'no user id'
    if len(user_id)>1:
        msg = 'received more than one user id'


