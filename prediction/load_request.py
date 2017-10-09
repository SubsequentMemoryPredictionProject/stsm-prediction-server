from sklearn.externals import joblib
import os
import sys
import ast
import pymysql.connections
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from DB.db_access import get_signals
from DB.db_access import choose_signals
from model_evaluation.validation_report import create_user_query
from logger import Logger

logger = Logger().get_logger()


def prediction_request_signals(request,conn):
    msg = 'success'
    logger.info('In prediction_request_signals')
    logger.info("User request: %s" % request)

    prediction_details = create_user_query(request)
    request_signals =choose_signals(conn,1,256,prediction_details,'user_data')
    return request_signals





