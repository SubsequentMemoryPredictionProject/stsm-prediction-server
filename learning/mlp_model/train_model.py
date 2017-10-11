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


logger = Logger().get_logger()


def train_and_save(db, electrode, duration,layer=(100, 20),activation='identity',cross_val=5):
    mlp_model = MLPClassifier(verbose=False, hidden_layer_sizes=layer, activation=activation)
    multi_mlp_model = MultiOutputClassifier(mlp_model, n_jobs=1)
    # load data from db to train & test model
    X = choose_signals(db,electrode,duration)
    logger.info('Finished getting signals for model training. size -%s' % str(np.shape(X)))
    Y = get_results(db)
    logger.info('Finished getting results for model training. size -%s' % str(np.shape(Y)))
    trained_model = cross_validation(X, Y, multi_mlp_model, cross_val)
    # save trained model
    joblib.dump(trained_model, 'trained_model2.pkl')
    return


def main():
    try:
        conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                               , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
        logger.info('Start model training')
        train_and_save(conn,4, 256, cross_val=7)
    except:
        logger.error('Error in training model - %s' % str(sys.exc_info()))


if __name__ == '__main__':
    main()
