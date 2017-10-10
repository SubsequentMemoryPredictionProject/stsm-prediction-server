from sklearn.model_selection import train_test_split
import sys, csv
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
NUM_RESULTS = 6

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from logger import Logger
from model_evaluation.test_model import evaluate_model

logger = Logger().get_logger()


def cross_validation(X, Y, model, k=5):
    scaler = StandardScaler(copy=False)
    precision = k*[NUM_RESULTS*[0]]
    precision_neg = k*[NUM_RESULTS*[0]]
    recall = k*[NUM_RESULTS*[0]]
    recall_neg = k*[NUM_RESULTS*[0]]
    f1 = k*[NUM_RESULTS*[0]]
    f1_neg = k*[NUM_RESULTS*[0]]
    for i in range(k):
        # split data to training and testing set
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,random_state=i)
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
    # report model scores
    cross_val_score(precision, recall, f1, precision_neg, recall_neg, f1_neg)
    return model


# calculate cross-validation score & save to file
def cross_val_score(precision, recall, f1, precision_neg, recall_neg, f1_neg):
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
        writer.writerow(['Remember ' + name + ' score:',])
        writer.writerow(np.mean(score, axis=0))
    for name, score in zip(names, forget_scores):
        logger.info('Forget - %s score = %s' % (name, np.mean(score, axis=0)))
        writer.writerow(['Forget ' + name + ' score:',])
        writer.writerow(np.mean(score, axis=0))
    file.close()
    return
