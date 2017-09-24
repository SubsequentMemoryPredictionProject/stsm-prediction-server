from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import numpy as np
import gc
from sklearn.model_selection import RandomizedSearchCV




PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from DB.db_access import get_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model
from model_evaluation.test_model import prescision_score
from model_evaluation.test_model import recall_score
from model_evaluation.test_model import f1_score




try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    param_distributions = {'estimator__hidden_layer_sizes': [(100,),(100,100),(120,120,120,120),(269,269)],'estimator__activation':
                           ['identity', 'logistic', 'tanh', 'relu'],'estimator__solver':['sgd','adam'],'estimator__alpha':
                           [0.0001,0.00001,0.001],'estimator__batch_size':['auto',100,50],'estimator__learning_rate':['constant',
                           'invscaling', 'adaptive'] ,'estimator__max_iter':[200,100],'estimator__shuffle':[True,False],'estimator__tol':[1e-4,1e-5],
                           'estimator__verbose':[False],'estimator__verbose':[True],'estimator__early_stopping':[True,False],
                           'estimator__beta_1':[0.9,0.7,0.5],'estimator__beta_2':[0.999,0.799,0.599],
                           'estimator__epsilon':[1e-8,1e-7]}
    mlp_model = MLPClassifier()
    mlp_multi_model = MultiOutputClassifier(mlp_model)
    clf = RandomizedSearchCV(estimator=mlp_multi_model,param_distributions=param_distributions,verbose=5,cv=2,
                             scoring=f1_score,n_iter=5)
    # load data to train & test model
    X = get_signals(conn)
    print('finished -  get data')
    Y = get_results(conn)
    print('finished - get results ')
    conn.close()
    print(sys.getsizeof(X))
    print(np.shape(X))
    print(np.shape(Y))

    # split data to training and testing set
    X_train, X_test, \
        Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    del X
    del Y
    gc.collect()
    clf.fit(X_train, Y_train)
    print(clf.cv_results_)
    print(clf.best_score_)
    print(clf.best_params_)
    print('finished model fit')

except:
    print(sys.exc_info()[0])
    raise
