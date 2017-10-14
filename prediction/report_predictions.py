import sys
from DB.db_access import insert_data
from DB.db_access import get_data
from model_evaluation.validation_report import create_user_query
from logger import Logger
from stsm_prediction_model.error_handling import UserRequestError
from stsm_prediction_model.error_handling import DBError


logger = Logger().get_logger()


# insert predictions to DB
def predictions_db(predictions, request, db):
    try:
        logger.info('Deleting duplicate predictions')
        delete_predictions_db(db, request)
    except (DBError, UserRequestError):
        raise
    user_id = request['user_id']
    insert_result = 0
    subjects_words = request['subjects_and_word_ids']
    query = 'INSERT INTO `untagged_predictions` (user_id,subject_id,word_id,stm,stm_confidence_level,stm_remember_know'\
            ',ltm,ltm_confidence_level,ltm_remember_know) VALUES'
    for i in subjects_words:
        for j in range(len(subjects_words[i])):
            request_details = '(' + str(user_id) + ',' + str(i)+',' + str(subjects_words[i][j])
            result_details = ',' + str(predictions[insert_result][0]) + ',' + str(predictions[insert_result][1])\
                + ',' + str(predictions[insert_result][2]) + ',' + str(predictions[insert_result][3])\
                + ',' + str(predictions[insert_result][4]) + ',' + str(predictions[insert_result][5]) + '),'
            query = query + request_details + result_details
            insert_result += 1
    query = query[:-1] + ';'
    try:
        insert_data(db, query)
    except DBError as err:
        raise DBError('Failed updating predictions to db - %s' % err.msg, err.code, str(sys.exc_info()))
    return


def delete_predictions_db(db, request):
    logger.info('Prediction table size before delete - %s'
                % str(get_data(db, 'SELECT count(row_count()) FROM untagged_predictions')))
    try:
        user_query = create_user_query(request)
    except:
        raise UserRequestError('Error in user request - creating SQL query', 6000, str(sys.exc_info()))
    query = 'DELETE FROM untagged_predictions WHERE'
    query = query + user_query
    try:
        insert_data(db, query)
        logger.info('Duplicate predictions deleted successfully.'
                    ' prediction table size - %s' %
                    str(get_data(db, 'SELECT count(row_count()) FROM untagged_predictions')))
    except DBError as err:
        raise DBError('Failed deleting duplicate predictions from '
                      'untagged prediction table - %s' % err.msg, err.code, str(sys.exc_info()))
    return


