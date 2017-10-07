from sklearn import metrics
import numpy as np
NUM_RESULTS = 6


def evaluate_model(y_true, y_pred):
    precision_score_forget, precision_score_remember = NUM_RESULTS*[0], NUM_RESULTS*[0]
    recall_score_forget,recall_score_remember = NUM_RESULTS*[0],NUM_RESULTS*[0]
    f1_score_forget,f1_score_remember = NUM_RESULTS*[0],NUM_RESULTS*[0]
    separate_predictions = separate_results(y_pred)
    separate_real_results = separate_results(y_true)
    for i in [1, 2, 4, 5]:
        precision_score_forget[i]=precision_score_remember[i] =\
            metrics.precision_score(separate_real_results[i], separate_predictions[i], average='weighted')
        recall_score_forget[i]=recall_score_remember[i] = \
            metrics.recall_score(separate_real_results[i], separate_predictions[i],average='weighted')
        f1_score_forget[i]=f1_score_remember[i] \
            = metrics.f1_score(separate_real_results[i], separate_predictions[i],average='weighted')
    for i in [0, 3]:
        precision_score_forget[i] = metrics.precision_score(separate_real_results[i],
                                                     separate_predictions[i],pos_label=0)
        precision_score_remember[i] = metrics.precision_score(separate_real_results[i],
                                                     separate_predictions[i])
        recall_score_forget[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i],pos_label=0)
        recall_score_remember[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i])
        f1_score_forget[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i],pos_label=0)
        f1_score_remember[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i])
    return precision_score_remember,recall_score_remember,f1_score_remember,precision_score_forget,\
           recall_score_forget,f1_score_forget


def separate_results(results):
    results_metrics = [[]*len(results) for i in range(NUM_RESULTS)]
    for row in results:
        for res in range(NUM_RESULTS):
            results_metrics[res].append(row[res])
    return results_metrics

