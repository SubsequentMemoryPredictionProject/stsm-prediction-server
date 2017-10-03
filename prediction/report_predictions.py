import csv
from DB.db_access import insert_data
import config as cfg
import ast


# create csv file with results
def predictions_file(results, request):
    filename = "results - subject_" + str(request[0][1]) + ".csv"
    file = open(filename, "w",newline='')
    writer = csv.writer(file, delimiter=',')
    for word in results:
        print(word)
        writer.writerow(word)
    file.close()
    return


# insert predictions to DB
def predictions_db(predictions, request,db):
    user_id = request['user_id']
    insert_result = 0
    #subjects_words = ast.literal_eval(request['subjectWords'])
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
            print(query)
            insert_result += 1
    query = query[:-1] + ';'
    insert_data(db, query)
    return
