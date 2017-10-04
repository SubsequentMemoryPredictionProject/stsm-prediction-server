from sklearn import metrics
import numpy as np
NUM_RESULTS = 6


def evaluate_model(y_true, y_pred):
    precision_score = NUM_RESULTS*[0]
    recall_score = NUM_RESULTS*[0]
    f1_score = NUM_RESULTS*[0]
    separate_predictions = separate_results(y_pred)
    separate_real_results = separate_results(y_true)
    for i in [1, 2, 4, 5]:
        precision_score[i] = metrics.precision_score(separate_real_results[i], separate_predictions[i], average='weighted')
        recall_score[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i],average='weighted')
        f1_score[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i],average='weighted')
    for i in [0, 3]:
        precision_score[i] = metrics.precision_score(separate_real_results[i],
                                                     separate_predictions[i],pos_label=0)
        recall_score[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i],pos_label=0)
        f1_score[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i],pos_label=0)
    return precision_score,recall_score,f1_score


def separate_results(results):
    results_metrics = [[]*len(results) for i in range(NUM_RESULTS)]
    for row in results:
        for res in range(NUM_RESULTS):
            results_metrics[res].append(row[res])
    return results_metrics

