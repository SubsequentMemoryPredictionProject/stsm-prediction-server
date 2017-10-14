from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import pymysql.connections
import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.preprocessing import StandardScaler
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from DB.db_access import choose_signals
from DB.db_access import get_results
from learning.svm_model.train_model import train_and_save

from logger import Logger

try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    logger = Logger().get_logger()

    # parameters to try:
    C = [1e-2, 1]
    kernels = ['linear', 'poly']
    electrode = [1, 2, 3, 4]
    eeg_duration = [240, 256, 260]

    Y = get_results(conn)
    matrix =[]
    average_matrix =[]
    precision=5*[0]
    recall = 5*[0]
    f1 = 5*[0]
    not_remember_precision =5*[0]
    not_remember_recall = 5*[0]
    not_remember_f1 = 5*[0]
    scaler = StandardScaler(copy=False)

    for elec in electrode:
        for dur in eeg_duration:
            X = choose_signals(conn, elec, dur)
            for c in C:
                for kernel in kernels:
                    train_and_save(conn, elec, dur, kernel, c)

except:
    logger.error('Error - %s ' % str(sys.exc_info()[0]))
    raise
finally:
    conn.close()