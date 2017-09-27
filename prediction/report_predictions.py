import csv
from DB.db_access import insert_data
import config as cfg


# create csv file with results
def results_file(results, request):
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
    for i in range(0, len(request)):
        request_details = "user_id=" + str(request[i][0]) + " AND subject_id=" + str(request[i][1]) \
            + " AND word_id=" + str(request[i][2])
        print(request_details)
        result_details = "stm=" + str(results[i][0]) + ",stm_confidence_level=" + str(results[i][1])\
            + ",stm_remember_know=" + str(results[i][2]) + ",ltm=" + str(results[i][3])\
            + ",ltm_confidence_level=" + str(results[i][4]) + ",ltm_remember_know=" + str(results[i][5])
        query = "UPDATE `untagged_predictions` SET " + result_details + " WHERE " + request_details
        insert_data(db, query)
    return

