from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
import pymysql.connections
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
import gc


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from DB.db_access import get_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model

try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    svm_model = svm.SVC(verbose=True)
    scaler = StandardScaler(copy=False)
    # load data to train & test model
    X = get_signals(conn)
    print('finished -  get data')
    Y = get_results(conn)
    print('finished - get results ')
    conn.close()

    print(np.shape(X))
    print(np.shape(Y))
    # split data to training and testing set
    X_train, X_test, \
        Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    del X
    del Y
    gc.collect()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    multi_svm_model = MultiOutputClassifier(svm_model, n_jobs=1)
    multi_svm_model.fit(X_train, Y_train)
    print('finished model fit')

    evaluate_model(multi_svm_model, X_test, Y_test)

    # save trained model
    joblib.dump(multi_svm_model, 'svm_model.pkl')
except:
    print(sys.exc_info()[0])
    raise