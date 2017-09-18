import os
from flask import Flask, jsonify, request
import json
from logger import Logger
import config as cfg
import pymysql
import sys

app = Flask(__name__)


# TODO comments
@app.route('/stsm/algorithms/predict', methods=['GET'])
def predict():
    return json.dumps({'ok': True})


@app.route('/stsm/algorithms/validate', methods=['GET'])
def validate():
    return json.dumps({'ok': True})


if __name__ == '__main__':
    try:
        logger = Logger().get_logger()
        logger.info('Starting web server...')
        print(cfg.mysql)
        db = pymysql.connect(host=cfg.mysql['host'], passwd=cfg.mysql['password']
                             , port=cfg.mysql['port'], user=cfg.mysql['user'], db=cfg.mysql['database'])
        logger.info('connected to db')
        app.run(host='0.0.0.0', port=3100)
        db.close()
        logger.info('disconnected from db')
    except:
        logger.error('ERROR: %s', sys.exc_info()[0])
        db.close()
        logger.info('disconnected from db')
        raise
