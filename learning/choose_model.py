from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import numpy as np
import gc
from sklearn import svm


PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from DB.db_access import get_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model


try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    classifiers =[MLPClassifier(max_iter=200,verbose=True,activation='identity'),
                  MLPClassifier(max_iter=200,hidden_layer_sizes=(120,120,120),batch_size=100,verbose=True),
                  MLPClassifier(max_iter=200,hidden_layer_sizes=(269),verbose=True)]

    names = ['mlp_default','mlp_3_layer','mlp_changed_layers','mlp_solver_lbfgs']
    # load data to train & test model
    X = get_signals(conn)
    print('finished -  get data')
    Y = get_results(conn)
    print('finished - get results ')
    conn.close()
    print(sys.getsizeof(X))
    print(np.shape(X))
    print(np.shape(Y))
    #X = np.asarray(X,dtype=np.ndarray)

    # split data to training and testing set
    X_train, X_test, \
        Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    del X
    del Y
    gc.collect()
    for name,clf in zip(names,classifiers):
        multi_mlp_model = MultiOutputClassifier(clf, n_jobs=1)
        multi_mlp_model.fit(X_train, Y_train)
        print('finished model fit')
        print("evaluate",name)
        evaluate_model(multi_mlp_model,X_test,Y_test)
    # save trained model
    #joblib.dump(multi_mlp_model, 'mlp_model.pkl')
except:
    print(sys.exc_info()[0])
    raise
