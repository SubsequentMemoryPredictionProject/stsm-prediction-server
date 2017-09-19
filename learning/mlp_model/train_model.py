from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import json
import numpy as np


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg

from DB.db_access import get_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model


try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    mlp_model = MLPClassifier(max_iter=400)

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

    multi_mlp_model = MultiOutputClassifier(mlp_model, n_jobs=1)
    multi_mlp_model.fit(X_train, Y_train)
    print('finished model fit')
    predictions = multi_mlp_model.predict(X_test)

    evaluate_model(multi_mlp_model,X_test,Y_test)
    # save trained model
    joblib.dump(multi_mlp_model, 'mlp_model.pkl')
except:
    print(sys.exc_info()[0])
    raise
