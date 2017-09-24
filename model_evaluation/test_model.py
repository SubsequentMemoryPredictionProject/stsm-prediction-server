from sklearn import metrics
from sklearn.metrics import make_scorer

NUM_RESULTS = 6


def evaluate_model(model, features_test, results_test):
    predictions = model.predict(features_test)
    precision_score = NUM_RESULTS*[0]
    recall_score = NUM_RESULTS*[0]
    f1_score = NUM_RESULTS*[0]
    separate_predictions = separate_results(predictions)
    separate_real_results = separate_results(results_test)
    for i in [1, 2, 4, 5]:
        precision_score[i] = metrics.precision_score(separate_real_results[i], separate_predictions[i], average='weighted')
        recall_score[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i],average='weighted')
        f1_score[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i],average='weighted')
    for i in [0, 3]:
        precision_score[i] = metrics.precision_score(separate_real_results[i],
                                                     separate_predictions[i],pos_label=0)
        recall_score[i] = metrics.recall_score(separate_real_results[i], separate_predictions[i],pos_label=0)
        f1_score[i] = metrics.f1_score(separate_real_results[i], separate_predictions[i],pos_label=0)
    mean_squared_error_score = metrics.mean_squared_error(results_test, predictions)
    print(precision_score)
    print(recall_score)
    print(f1_score)
    print(mean_squared_error_score)

    return

def precision_score(results_test, predictions):
    separate_predictions = separate_results(predictions)
    separate_real_results = separate_results(results_test)
    precision_score = metrics.precision_score(separate_real_results[0],
                                              separate_predictions[0], pos_label=0)
    return precision_score

make_scorer(precision_score)

def recall_score(results_test,predictions):
    separate_predictions = separate_results(predictions)
    separate_real_results = separate_results(results_test)
    recall_score = metrics.recall_score(separate_real_results[0],
                                        separate_predictions[0], pos_label=0)
    return recall_score

def separate_results(results):
    results_metrics = [[]*len(results) for i in range(NUM_RESULTS)]
    for row in results:
        for res in range(NUM_RESULTS):
            results_metrics[res].append(row[res])
    return results_metrics
