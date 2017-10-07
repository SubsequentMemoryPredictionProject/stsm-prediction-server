from DB.db_access import insert_data
from DB.db_access import get_data
from model_evaluation.validation_report import create_user_query


# insert predictions to DB
def predictions_db(predictions, request,db):
    delete_predictions_db(db, request)
    user_id = request['user_id']
    insert_result = 0
    subjects_words = request['subjects_and_word_ids']
    query = 'INSERT INTO `untagged_predictions` (user_id,subject_id,word_id,stm,stm_confidence_level,stm_remember_know' \
            ',ltm,ltm_confidence_level,ltm_remember_know) VALUES'
    for i in subjects_words:
        for j in range(len(subjects_words[i])):
            request_details = '(' + str(user_id) + ','+ str(i)+','+ str(subjects_words[i][j])
            result_details = ',' + str(predictions[insert_result][0]) + ',' + str(predictions[insert_result][1])\
            + ',' + str(predictions[insert_result][2]) + ',' + str(predictions[insert_result][3])\
            + ',' + str(predictions[insert_result][4]) + ',' + str(predictions[insert_result][5]) +'),'
            query = query + request_details + result_details
            insert_result += 1
    query = query[:-1] + ';'
    insert_data(db, query)
    return


def delete_predictions_db(db,request):
    print (get_data(db,'SELECT count(*) FROM untagged_predictions'))
    user_query  = create_user_query(request)
    query = 'DELETE FROM untagged_predictions WHERE'
    query = query + user_query
    insert_data(db,query)
    print (get_data(db,'SELECT count(*) FROM untagged_predictions'))
    return


