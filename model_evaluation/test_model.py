from sklearn import metrics
from stsm_prediction_model.error_handling import ModelError
from sklearn.externals import joblib
from model_evaluation.cross_validation import d_prime
import numpy as np
import pymysql.connections
import sys
import os
NUM_RESULTS = 6
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from logger import Logger
from DB.db_access import choose_signals
from DB.db_access import get_results


def evaluate_model(y_true, y_pred):
    precision_score_forget, precision_score_remember = NUM_RESULTS*[0], NUM_RESULTS*[0]
    recall_score_forget, recall_score_remember = NUM_RESULTS*[0], NUM_RESULTS*[0]
    f1_score_forget, f1_score_remember = NUM_RESULTS*[0], NUM_RESULTS*[0]
    separate_predictions = separate_results(y_pred)
    separate_real_results = separate_results(y_true)
    try:
        for i in [1, 2, 4, 5]:
            precision_score_forget[i] = precision_score_remember[i] =\
                metrics.precision_score(separate_real_results[i], separate_predictions[i], average='weighted')
            recall_score_forget[i] = recall_score_remember[i] = \
                metrics.recall_score(separate_real_results[i], separate_predictions[i], average='weighted')
            f1_score_forget[i] = f1_score_remember[i] \
                = metrics.f1_score(separate_real_results[i], separate_predictions[i], average='weighted')
        for i in [0, 3]:
            precision_score_forget[i] = metrics.precision_score(separate_real_results[i],
                                                                separate_predictions[i], pos_label=0)
            precision_score_remember[i] = metrics.precision_score(separate_real_results[i], separate_predictions[i])
            recall_score_forget[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i],pos_label=0)
            recall_score_remember[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i])
            f1_score_forget[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i],pos_label=0)
            f1_score_remember[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i])
    except:
        raise ModelError('Error in model evaluation', 1008, sys.exc_info()[1])
    return precision_score_remember, recall_score_remember, f1_score_remember, precision_score_forget,\
        recall_score_forget, f1_score_forget


def separate_results(results):
    results_metrics = [[]*len(results) for i in range(NUM_RESULTS)]
    for row in results:
        for res in range(NUM_RESULTS):
            results_metrics[res].append(row[res])
    return results_metrics


def main():
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                           , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
    model = joblib.load('trained_model.pkl')
    X = choose_signals(conn, 1, 256)
    Y = get_results(conn)
    prediction = model.predict(X)
    d_prime(Y, prediction, X, model)

if __name__ == '__main__':
    main()
