import ast
from DB.db_access import get_results


def validate_user_results(request,db):
    query = create_user_query(request)
    true_values = get_results(db,query,'data_set')
    #pred_values = get_results(db,query,'untagged_predictions')
    print(true_values)
    #print(pred_values)
    return

def create_user_query(request):
    print(request)
    user_id = request['user_id']
    print(user_id)
    query = 'AND ( '
    subjects_words = request['subjects_and_word_ids']
    print(subjects_words)
    for i in subjects_words:
        for j in range(len(subjects_words[i])):
            request_details = " (user_id=" + str(user_id) + " AND subject_id=" + str(i) \
                                 + " AND word_id=" + str(subjects_words[i][j]) + ') OR'
            query = query + request_details
    query = query[:-2] + ');'
    print(query)
    return query
