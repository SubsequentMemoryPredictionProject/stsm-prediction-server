from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import pymysql.connections
import sys, os
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
from model_evaluation.test_model import separate_results




try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

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
                    svm_model = svm.SVC(C=c,max_iter=5500,verbose=True,kernel=kernel)
                    multi_svm_model = MultiOutputClassifier(svm_model, n_jobs=1)
                    print(np.shape(X))
                    print(np.shape(Y))
                    # cross validation
                    for i in range(5):
                        X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=i)
                        scaler.fit(X_train)
                        scaler.transform(X_train)
                        scaler.transform(X_test)
                        multi_svm_model.fit(X_train, Y_train)
                        print('finished model fit')
                        y_pred = separate_results(multi_svm_model.predict(X_test))[0]
                        y_true = separate_results(Y_test)[0]
                        precision[i] = metrics.precision_score(y_true,y_pred)
                        recall[i] = metrics.recall_score(y_true,y_pred)
                        f1[i] = metrics.f1_score(y_true,y_pred)
                        not_remember_precision[i] = metrics.precision_score(y_true, y_pred, pos_label=0)
                        not_remember_recall[i] = metrics.recall_score(y_true, y_pred, pos_label=0)
                        not_remember_f1[i] = metrics.f1_score(y_true,y_pred,pos_label=0)
                        matrix = confusion_matrix(y_true, y_pred)
                        normalize_matrix = matrix / matrix.astype(np.float).sum(axis=1, keepdims=True)
                        average_matrix.append(normalize_matrix)
                        matrix =[]
                    print("params: elctrode - %d, duration = %d, C = %s, kernel = %s" % (elec, dur, c,kernel))
                    print("precision = %f"%(np.mean(precision)))
                    print("recll = %f"%(np.mean(recall)))
                    print("f1 = %f" % (np.mean(f1)))
                    print("negative lable precision = %f" % (np.mean(not_remember_precision)))
                    print("negative lable recll = %f" % (np.mean(not_remember_recall)))
                    print("negative lable f1 = %f" % (np.mean(not_remember_f1)))
                    print("confusion matrix =")
                    print(np.mean(average_matrix,axis=0))
                    average_matrix =[]
except:
    print(sys.exc_info()[0])
    raise
finally:
    conn.close()
