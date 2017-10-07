from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys,csv
import os
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix




PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg

from DB.db_access import choose_signals
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model
from model_evaluation.test_model import separate_results


try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    mlp_model = MLPClassifier(verbose=True,hidden_layer_sizes=(100,20),activation='identity')

    scaler = StandardScaler(copy=False)
    # load data to train & test model
    X = choose_signals(conn, 1, 256)
    print('finished -  get data')
    Y = get_results(conn)
    print('finished - get results ')
    conn.close()
    print(np.shape(X))
    print(np.shape(Y))

    # split data to training and testing set
    X_train, X_test, \
        Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    del X
    del Y
    gc.collect()
    X_train = scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    multi_mlp_model = MultiOutputClassifier(mlp_model, n_jobs=1)
    multi_mlp_model.fit(X_train, Y_train)
    print('finished model fit')
    print(np.shape(X_train))
    print(np.shape(X_test))
    print(np.shape(Y_train))
    print(np.shape(Y_test))

    Y_pred = multi_mlp_model.predict(X_test)
    prob = multi_mlp_model.predict_proba(X_test)
    prob_stm = prob[0]
    pred_stm = separate_results(Y_pred)[0]
    true_stm = separate_results(Y_test)[0]
    matrix = confusion_matrix(true_stm,pred_stm)
    normalized_matrix = matrix / matrix.astype(np.float).sum(axis=1, keepdims=True)
    print(prob_stm)
    print(pred_stm)
    print(true_stm)
    filename = 'bestModelResultsStm2.csv'
    file = open(filename, "w", newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(pred_stm)
    writer.writerow(true_stm)
    writer.writerow(prob_stm)
    prec,recall,f1,neg_prec,neg_recall,neg_f1 = evaluate_model(Y_test,Y_pred)
    writer.writerow(['Remember - precision,recall,f1:',prec[0],recall[0],f1[0]])
    writer.writerow(['Forget - precision,recall,f1:',neg_prec[0],neg_recall[0],neg_f1[0]])
    writer.writerow(['confusion-matrix (total=%d)'%(len(true_stm))])
    writer.writerow(matrix[0])
    writer.writerow(matrix[1])
    writer.writrow(['Normalized confusion matrix:'])
    writer.writerow(normalized_matrix[0])
    writer.writerow(normalized_matrix[1])
    file.close()

    # save trained model
    joblib.dump(multi_mlp_model, 'mlp_model.pkl')
except:
    print(sys.exc_info()[0])
    raise

