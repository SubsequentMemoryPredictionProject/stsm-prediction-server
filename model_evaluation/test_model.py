from sklearn import metrics
NUM_RESULTS = 6


def evaluate_model(model, features_test, results_test):
    predictions = model.predict(features_test)
    precision_score, recall_score, f1_score = [], [], []
    separate_predictions = separate_results(predictions)
    separate_real_results = separate_results(results_test)
    for i in range(NUM_RESULTS):
        precision_score.append(metrics.precision_score(separate_real_results[i],
                                                       separate_predictions[i],average='micro'))
        recall_score.append(metrics.recall_score(separate_real_results[i], separate_predictions[i], average='micro'))
        f1_score.append(metrics.f1_score(separate_real_results[i], separate_predictions[i],average='micro'))
    mean_squared_error_score = metrics.mean_squared_error(results_test, predictions)
    print(precision_score)
    print(recall_score)
    print(f1_score)
    print(mean_squared_error_score)
    return


def separate_results(results):
    results_metrics = [[]*len(results) for i in range(NUM_RESULTS)]
    for row in results:
        for res in range(NUM_RESULTS):
            results_metrics[res].append(row[res])
    return results_metrics
