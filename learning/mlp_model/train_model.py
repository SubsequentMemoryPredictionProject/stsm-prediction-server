from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
import pymysql.connections
import sys
import os
import numpy as np
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from logger import Logger
from DB.db_access import choose_signals
from DB.db_access import get_results
from learning.cross_validation import cross_validation
from stsm_prediction_model.error_handling import ModelError
from stsm_prediction_model.error_handling import DBError


logger = Logger().get_logger()


def train_and_save(db, electrode, duration, layer=(100, 20), activation='identity', cross_val=5,
                   learning_rate='constant'):
    try:
        mlp_model = MLPClassifier(verbose=True, hidden_layer_sizes=layer, activation=activation,
                                  solver='sgd', learning_rate=learning_rate, max_iter=500)
        multi_mlp_model = MultiOutputClassifier(mlp_model, n_jobs=1)
    except:
        raise ModelError('Error in creating MLP model', 1014, sys.exc_info()[1])
    # load data from db to train & test model
    try:
        X = choose_signals(db, electrode, duration)
        logger.info('Finished getting signals for model training. size -%s' % str(np.shape(X)))
        Y = get_results(db)
        logger.info('Finished getting results for model training. size -%s' % str(np.shape(Y)))
    except DBError as err:
        raise DBError('Failed getting signals/results - %s' % err.msg, 1015, sys.exc_info()[1])
    logger.info('Results using params: electrode =%d, duration = %d, layers = %s, activation = %s, learning_rate = %s' %
                (electrode, duration, str(layer), activation, learning_rate))
    trained_model = cross_validation(X, Y, multi_mlp_model, cross_val)
    # save trained model
    try:
        joblib.dump(trained_model, 'trained_model.pkl')
    except:
        raise ModelError('Error saving trained model', 1016, sys.exc_info()[1])
    return


def main():
    try:
        conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                               , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
        logger.info('Start model training')
        train_and_save(conn, 1, 256, cross_val=1, learning_rate='invscaling')
    except (ModelError,DBError):
        logger.error('Error in training model - %s' % str(sys.exc_info()[1]))


if __name__ == '__main__':
    main()
