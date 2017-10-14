import os
import sys
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from logger import Logger
from model_evaluation.test_model import evaluate_model
from model_evaluation.test_model import separate_results
from stsm_prediction_model.error_handling import ModelError


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
        writer.writerow([val, ])
        writer.writerow(res)
    for val, res in zip(pred_vals, pred_separate):
        writer.writerow([val, ])
        writer.writerow(res)
    proba_vals = ['Stm:', 'Stm confidence:', 'Stm Remember/Know:', 'Ltm:', 'Ltm confidence:', 'Ltm Remember/Know:']
    try:
        probability = model.predict_proba(x_test)
    except:
        raise ModelError('Error in model evaluation -  predict_probability', 4003, str(sys.exc_info()))
    writer.writerow(['Predict probability', ])
    for val, prob in zip(proba_vals, probability):
        writer.writerow([val, ])
        writer.writerow(prob)
    file.close()
    return


def confusion_matrix_file(y_true, y_pred):
    separate_true = separate_results(y_true)
    separate_pred = separate_results(y_pred)
    filename = 'ConfusionMatrix.csv'
    try:
        matrix_stm = confusion_matrix(separate_true[0], separate_pred[0])
        normalize_matrix_stm = matrix_stm / matrix_stm.astype(np.float).sum(axis=1, keepdims=True)
        matrix_ltm = confusion_matrix(separate_true[3], separate_pred[3])
        normalize_matrix_ltm = matrix_ltm / matrix_ltm.astype(np.float).sum(axis=1, keepdims=True)
    except:
        raise ModelError('Error in model evaluation - confusion matrix', 4003, str(sys.exc_info()))
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
