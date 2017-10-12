import sys
import csv
from DB.db_access import get_results
from model_evaluation.test_model import evaluate_model
from prediction.load_request import create_user_query
from stsm_prediction_model.error_handling import ModelError
from stsm_prediction_model.error_handling import DBError
from stsm_prediction_model.error_handling import UserRequestError
from logger import Logger
import numpy as np

logger = Logger().get_logger()


def validate_user_results(request, db):
    try:
        query = create_user_query(request)
    except:
        raise UserRequestError('Error in user request - failed to create SQL query', 1004, sys.exc_info()[1])
    try:
        true_values = get_results(db, query, 'user_data')
        logger.info('Finished getting user results. size - %s' % str(np.shape(true_values)))
        pred_values = get_results(db, query, 'untagged_predictions')
        logger.info('Finished getting predictions. size - %s' % str(np.shape(pred_values)))
        if not (np.size(true_values) and np.size(pred_values)):
            raise DBError('Results/predictions doesnt exist', 1016, sys.exc_info()[1])
    except DBError as err:
        raise DBError('Failed getting user results/predictions for validation - %s' % err.msg, err.code, err.error)
    try:
        precision_remember, recall_remember, f1_remember, precision_forget, recall_forget, f1_forget\
            = evaluate_model(true_values, pred_values)
    except ModelError:
        raise
    return model_evaluation_file(precision_remember, recall_remember, f1_remember,precision_forget, recall_forget,
                                 f1_forget)


# create csv file with results scores
def model_evaluation_file(precision_remember, recall_remember, f1_remember, precision_forget, recall_forget, f1_forget):
    func_name = ['Precision-remember:', 'Recall-remember:', 'F1-remember:', 'Precision-forget:', 'Recall-forget:',
                 'F1-forget:']
    model_scores = [precision_remember, recall_remember, f1_remember, precision_forget, recall_forget, f1_forget]
    filename = 'validationScores.csv'
    try:
        file = open(filename, 'w', newline='')
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['', 'Stm', 'Stm confidence level', 'Stm remember/know', 'Ltm', 'Ltm confidence level',
                         'Ltm remember/know'])
        for name, score in zip(func_name, model_scores):
            score.insert(0, name)
            writer.writerow(score)
        file.close()
    except:
        raise ModelError('Failed creating validation file', 1009, sys.exc_info()[1])
    return filename
