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


# insert results to DB
def results_db(results, request,db):
    user_id = request['user_id']
    insert_result=0
    subjects_words = ast.literal_eval(request['subjectWords'])
    for i in subjects_words:
        for j in range(len(subjects_words[i])):
            request_details = str(user_id) + ",subject_id=" + str(i) \
                                 + ",word_id=" + str(subjects_words[i][j])
            result_details = ",stm=" + str(results[insert_result][0]) + ",stm_confidence_level=" + str(results[insert_result][1])\
            + ",stm_remember_know=" + str(results[insert_result][2]) + ",ltm=" + str(results[insert_result][3])\
            + ",ltm_confidence_level=" + str(results[insert_result][4]) + ",ltm_remember_know=" + str(results[insert_result][5]) + ");"
            query = "INSERT INTO `untagged_predictions` VALUES( " + request_details + result_details
            print(query)
            insert_data(db, query)
            insert_result+=1
        return
