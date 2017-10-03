from sklearn.externals import joblib
import os
import sys
import ast
import pymysql.connections
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from DB.db_access import get_signals
import config as cfg


def prediction_request(request,conn):
    print(request)
    request_signals =[]
    user_id = request['user_id']
    print(user_id)
    subjects_words = request['subjects_and_word_ids']
    print(subjects_words)
    for i in subjects_words:
        print(subjects_words[i])
        for j in range(len(subjects_words[i])):
            prediction_details = " AND user_id=" + str(user_id) + " AND subject_id=" + str(i)\
                         + " AND word_id=" + str(subjects_words[i][j])
            #TODO change table to user_data
            request_signals.extend(get_signals(conn,prediction_details,'data_set'))
    return request_signals
