from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler




PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg

from DB.db_access import get_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model


try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    classifiers =[MLPClassifier(max_iter=100,verbose=True,activation='tanh',alpha=1e-5,batch_size=100,epsilon=10e-7,
                                beta_2=0.599,hidden_layer_sizes=(100,100)),
                  MLPClassifier(max_iter=100,hidden_layer_sizes=(100,100),verbose=True),
                  MLPClassifier(max_iter=200,alpha=1e-5,verbose=True)]

    names = ['best','mlp_2_layer','mlp_changed_layers','mlp_changed_alpha']
    scaler = StandardScaler(copy=False)

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
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
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
