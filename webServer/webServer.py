import os
import sys
import ast
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)

from flask import Flask, jsonify, request
import json
from logger import Logger
import config as cfg
import pymysql

from stsm_prediction_model.stsm_model import StsmPredictionModel
app = Flask(__name__)
stsm_model = None


# TODO comments

@app.route('/stsm/algorithms/predict/', methods=['POST'])
def predict():
    try:
        print(request.get_json())
        stsm_model.evaluate(request.get_json())
        return json.dumps({'Success': True})
    except:
        return json.dumps({'Success':False})


@app.route('/stsm/algorithms/validate/')
def validate():
    return jsonify(stsm_model.evaluate(request.form.getlist('arr', type=int)))
    return json.dumps({'ook': True})


if __name__ == '__main__':
    try:
        logger = Logger().get_logger()
        logger.info('Starting web server...')

        stsm_model = StsmPredictionModel()
        stsm_model.load_model()
        stsm_model.connect()
        app.run( host='0.0.0.0',port=3100)

    except:
        logger.error('ERROR: %s' %(sys.exc_info()[0]))
        stsm_model.disconnect()
        logger.info('disconnected from db')
        raise
    finally:
        stsm_model.disconnect()
