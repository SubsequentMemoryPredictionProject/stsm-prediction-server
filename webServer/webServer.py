import os
import sys
import ast

PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)

from flask import Flask, jsonify, request, send_file
import json
from logger import Logger
import config as cfg
import pymysql

from stsm_prediction_model.stsm_model import StsmPredictionModel

app = Flask(__name__)
stsm_model = None


# TODO comments

@app.route('/stsm/algorithms/predict', methods=['POST'])
def predict():
    try:
        print('request.get_json()', request.get_json())
        stsm_model.evaluate(request.get_json())
        return json.dumps({'msg': 'Prediction process was done successfully', 'success': True})
    except:
        logger.error('ERROR: %s'%str(sys.exc_info()))
        return json.dumps({'msg': 'Prediction process failed', 'success': False})


@app.route('/stsm/algorithms/validate',methods=['POST'])
def validate():
    try:
        print('request.get_json()', request.get_json())
        validation_file = stsm_model.validate(request.get_json())
        return send_file (validation_file,as_attachment=True), \
               json.dumps({'msg': 'Validation process was done successfully', 'success': True})
    except:
        logger.error('Error : %s'%str(sys.exc_info()))
        return json.dumps({'msg': 'Validation process failed', 'success': False})


if __name__ == '__main__':
    try:
        logger = Logger().get_logger()
        logger.info('Starting web server...')
        stsm_model = StsmPredictionModel(logger)
        stsm_model.load_model()
        stsm_model.connect()
        app.run(host='0.0.0.0', port=3100)

    except:
        logger.error('ERROR: %s' % str(sys.exc_info()))
        stsm_model.disconnect()
        logger.info('disconnected from db')
        raise
    finally:
        stsm_model.disconnect()
