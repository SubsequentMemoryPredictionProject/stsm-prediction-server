import ast
import csv
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model


def validate_user_results(request,db):
    query = create_user_query(request)
    true_values = get_results(db, query, 'data_set')
    pred_values = get_results(db, query, 'untagged_predictions')
    print(true_values)
    print(pred_values)
    precision, recall, f1 = evaluate_model(true_values,pred_values)
    return model_evaluation_file(precision, recall, f1)


def create_user_query(request):
    print(request)
    user_id = request['user_id']
    print(user_id)
    query = ' ('
    subjects_words = request['subjects_and_word_ids']
    print(subjects_words)
    for i in subjects_words:
        for j in range(len(subjects_words[i])):
            request_details = "(user_id=" + str(user_id) + " AND subject_id=" + str(i) \
                                 + " AND word_id=" + str(subjects_words[i][j]) + ') OR'
            query = query + request_details
    query = query[:-2] + ');'
    return query


# create csv file with results scores
def model_evaluation_file( precision,recall,f1):
    print('in create file')
    func_name = ['precision:','recall:','F1:']
    model_scores = [precision,recall,f1]

    filename = "results validation" + ".csv"
    file = open(filename, "w",newline='')
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["","stm","stm confidence level","stm remember/know","ltm","ltm confidence level","ltm remember/know"])
    for name,score in zip(func_name,model_scores):
        score.insert(0,name)
        writer.writerow(score)
    file.close()
    return filename