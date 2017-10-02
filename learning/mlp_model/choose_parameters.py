from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix




PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from DB.db_access import choose_signals
from DB.db_access import get_results
from model_evaluation.test_model import separate_results




try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    # parameters to try:
    layer_size = [(100, 20), (100, 80, 20)]
    activation = ['relu', 'tanh', 'identity']
    electrode = [1, 2, 3, 4]
    eeg_duration = [240, 256, 260]

    Y = get_results(conn)
    matrix =[]
    average_matrix =[]
    precision=5*[0]
    recall = 5*[0]
    not_remember_precision =5*[0]
    not_remember_recall = 5*[0]

    for elec in electrode:
        for dur in eeg_duration:
            X = choose_signals(conn, elec, dur)
            for lyer in layer_size:
                for func in activation:
                    mlp_model = MLPClassifier(hidden_layer_sizes=lyer,activation=func)
                    mlp_multi_model = MultiOutputClassifier(mlp_model)
                    print(np.shape(X))
                    print(np.shape(Y))
                    # cross validation
                    for i in range(5):
                        X_train, X_test, \
                        Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=i)
                        mlp_multi_model.fit(X_train, Y_train)
                        print('finished model fit')
                        y_pred = separate_results(mlp_multi_model.predict(X_test))[0]
                        y_true = separate_results(Y_test)[0]
                        precision[i]= metrics.precision_score(y_true,y_pred)
                        recall[i] = metrics.recall_score(y_true,y_pred)
                        not_remember_precision[i] = metrics.precision_score(y_true,y_pred,pos_label=0)
                        not_remember_recall[i] = metrics.recall_score(y_true,y_pred,pos_label=0)
                        matrix = confusion_matrix(y_true,y_pred)
                        normelaize_matrix = matrix/matrix.astype(np.float).sum(axis=1)
                        average_matrix.append(normelaize_matrix)
                        matrix =[]
                    print("params: elctrode - %d, duration = %d, layers = %s, activation = %s"%(elec,dur,lyer,func))
                    #print(confusion_matrix(separate_results(Y_test)[0], separate_results(y_pred)[0]))
                    print("precision = %f"%(np.mean(precision)))
                    print("recll = %f"%(np.mean(recall)))
                    print("negative lable precision = %f" % (np.mean(not_remember_precision)))
                    print("negative lable recll = %f" % (np.mean(not_remember_recall)))
                    print("normelaized confusion matrix average = %s"%(np.mean(average_matrix,axis=0)))
                    average_matrix =[]
    conn.close()


except:
    print(sys.exc_info()[0])
    conn.close()
    raise
