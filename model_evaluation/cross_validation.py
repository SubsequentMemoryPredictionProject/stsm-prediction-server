from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys, csv
import os
import numpy as np
from sklearn.preprocessing import Normalizer
NUM_RESULTS = 6

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from logger import Logger
from model_evaluation.test_model import evaluate_model
from model_evaluation.test_model import separate_results
from stsm_prediction_model.error_handling import ModelError


logger = Logger().get_logger()


def cross_validation(X, Y, model, k=5, scale=False):
    scaler = Normalizer(copy=False)
    precision, precision_neg = k*[NUM_RESULTS*[0]], k*[NUM_RESULTS*[0]]
    recall, recall_neg = k*[NUM_RESULTS*[0]], k*[NUM_RESULTS*[0]]
    f1, f1_neg = k*[NUM_RESULTS*[0]], k*[NUM_RESULTS*[0]]
    average_matrix = []
    average_matrix2 = []
    for i in range(k):
        # split data to training and testing set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=i)
        if scale:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        model.fit(X_train,Y_train)
        logger.info('Cross-validation fold - %d :finished model fit' % (i+1))
        Y_pred = model.predict(X_test)
        logger.info('Cross-validation fold - %d :finished prediction of test samples' % (i+1))
        remember_prec, remember_recall, remember_f1,\
            forget_prec, forget_recall, forget_f1 = evaluate_model(Y_test, Y_pred)
        precision[i] = remember_prec
        recall[i] = remember_recall
        f1[i] = remember_f1
        precision_neg[i] = forget_prec
        recall_neg[i] = forget_recall
        f1_neg[i] = forget_f1
        matrix_stm = confusion_matrix(separate_results(Y_test)[0], separate_results(Y_pred)[0])
        normalize_matrix_stm = matrix_stm / matrix_stm.astype(np.float).sum(axis=1, keepdims=True)
        average_matrix.append(normalize_matrix_stm)
        matrix_ltm = confusion_matrix(separate_results(Y_test)[3], separate_results(Y_pred)[3])
        normalize_matrix_ltm = matrix_ltm / matrix_ltm.astype(np.float).sum(axis=1, keepdims=True)
        average_matrix2.append(normalize_matrix_ltm)
    # report model scores
    cross_val_score(precision, recall, f1, precision_neg, recall_neg, f1_neg, average_matrix, average_matrix2)
    d_prime(Y_test, Y_pred, X_test, model)
    return model


# calculate cross-validation score & save to file
def cross_val_score(precision, recall, f1, precision_neg, recall_neg, f1_neg, stm=None, ltm=None):
    remember_scores = [precision, recall, f1]
    forget_scores = [precision_neg, recall_neg, f1_neg]
    filename = 'bestModelResults.csv'
    file = open(filename, "w", newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Stm", "Stm confidence level", "Stm remember/know", "Ltm",
                    "Ltm confidence level", "Ltm remember/know"])
    names = ['precision', 'recall', 'f1']
    for name, score in zip(names, remember_scores):
        logger.info('Remember - %s score = %s' % (name, np.mean(score, axis=0)))
        writer.writerow(['Remember ' + name + ' score:', ])
        writer.writerow(np.mean(score, axis=0))
    for name, score in zip(names, forget_scores):
        logger.info('Forget - %s score = %s' % (name, np.mean(score, axis=0)))
        writer.writerow(['Forget ' + name + ' score:', ])
        writer.writerow(np.mean(score, axis=0))
    if stm:
        logger.info('Normalized confusion matrix - Stm: %s' % str(np.mean(stm, axis=0)))
        writer.writerow(['Normalized confusion matrix - Stm:',])
        writer.writerow(stm)
    if ltm:
        logger.info('Normalized confusion matrix - Ltm: %s' % str(np.mean(ltm, axis=0)))
        writer.writerow(['Normalized confusion matrix - Ltm:', ])
        writer.writerow(ltm)
    file.close()
    return


def d_prime(y_true, y_pred, x_test, model):
    true_separate = separate_results(y_true)
    pred_separate = separate_results(y_pred)
    filename = 'resultsForDPrime.csv'
    file = open(filename, "w", newline='')
    writer = csv.writer(file, delimiter=',')
    true_vals = ['True -Stm:', 'True - Stm confidence:', 'True - Stm Remember/Know:',
                 'True - Ltm:', 'True - Ltm confidence:', 'True - Ltm Remember/Know:']
    pred_vals = ['Pred - Stm:', 'Pred - Stm confidence:', 'Pred - Stm Remember/Know:',
                 'Pred - Ltm:', 'Pred - Ltm confidence:', 'Pred - Ltm Remember/Know:']
    for val, res in zip(true_vals, true_separate):
        writer.writerow([val,])
        writer.writerow(res)
    for val, res in zip(pred_vals, pred_separate):
        writer.writerow([val,])
        writer.writerow(res)
    proba_vals = ['Stm:', 'Stm confidence:', 'Stm Remember/Know:', 'Ltm:', 'Ltm confidence:', 'Ltm Remember/Know:']
    probability = model.predict_proba(x_test)
    writer.writerow(['Predict probability', ])
    for val, prob in zip(proba_vals, probability):
        writer.writerow([val, ])
        writer.writerow(prob)
    file.close()
    return


def confusion_matrix_file(y_true, y_pred):
    print('true size = ',np.shape(y_pred))
    print(' pred size = ',np.shape(y_true))
    separate_true = separate_results(y_true)
    separate_pred = separate_results(y_pred)
    filename = 'ConfusionMatrix.csv'
    matrix_stm = confusion_matrix(separate_true[0], separate_pred[0])
    normalize_matrix_stm = matrix_stm / matrix_stm.astype(np.float).sum(axis=1, keepdims=True)
    matrix_ltm = confusion_matrix(separate_true[3], separate_pred[3])
    normalize_matrix_ltm = matrix_ltm / matrix_ltm.astype(np.float).sum(axis=1, keepdims=True)
    file = open(filename, "w", newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['Confusion matrix - Stm (total =%d):' % len(y_true), ])
    writer.writerow(matrix_stm)
    writer.writerow(['Normalized confusion matrix - Stm:', ])
    writer.writerow(normalize_matrix_stm)
    writer.writerow(['Confusion matrix - Ltm (total =%d):' % len(y_true), ])
    writer.writerow(matrix_ltm)
    writer.writerow(['Normalized confusion matrix - Ltm:', ])
    writer.writerow(normalize_matrix_ltm)
    file.close()
    return



