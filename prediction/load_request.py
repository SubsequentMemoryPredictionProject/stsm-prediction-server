from sklearn.externals import joblib
import os
import sys
import ast
import pymysql.connections
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from DB.db_access import choose_signals
from model_evaluation.validation_report import create_user_query
from stsm_prediction_model.error_handling import DBError
from stsm_prediction_model.error_handling import UserRequestError
from logger import Logger

logger = Logger().get_logger()


def prediction_request_signals(request, conn):
    logger.info('In prediction_request_signals')
    logger.info("User request: %s" % request)
    try:
        prediction_details = create_user_query(request)
    except:
        raise UserRequestError('Error in user request - failed to create SQL query', 1004, sys.exc_info()[1])
    try:
        request_signals = choose_signals(conn, 1, 256, prediction_details, 'user_data')
        logger.info('Successful in getting eeg signals for user request')
    except:
        raise DBError('Failed getting eeg signals for user request', 1003, sys.exc_info()[1])
    return request_signals





