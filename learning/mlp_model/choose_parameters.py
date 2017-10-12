import pymysql.connections
import sys, os

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
import config as cfg
from learning.mlp_model.train_model import train_and_save
from logger import Logger

logger = Logger().get_logger()
try:
    conn = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                     , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])

    # parameters to try:
    layer_size = [(100, 20), (100, 80, 20)]
    activation = ['relu', 'tanh', 'identity']
    electrode = [1, 2, 3, 4]
    eeg_duration = [240, 256, 260]
    learning_rate = ['constant', 'invscaling', 'adaptive']

    for elec in electrode:
        for dur in eeg_duration:
            for lyer in layer_size:
                for func in activation:
                    for rate in learning_rate:
                        train_and_save(conn, elec, dur, lyer, func, rate)

except:
    logger.error('Error - %s ' % str(sys.exc_info()[0]))
    raise
finally:
    conn.close()
