from sklearn.externals import joblib
import os
import sys
import ast
import pymysql.connections
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from DB.db_access import choose_signals
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
    except DBError as err:
        raise DBError('Failed getting eeg signals for user request - %s' % err.msg, err.code, sys.exc_info()[1])
    return request_signals


def create_user_query(request):
    user_id = request['user_id']
    query = ' ('
    subjects_words = request['subjects_and_word_ids']
    for i in subjects_words:
        for j in range(len(subjects_words[i])):
            request_details = "(user_id=" + str(user_id) + " AND subject_id=" + str(i) \
                                 + " AND word_id=" + str(subjects_words[i][j]) + ') OR'
            query = query + request_details
    query = query[:-2] + ');'
    return query





