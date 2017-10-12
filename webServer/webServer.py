import os
import sys
import signal
PROJECT_ROOT = os.path.abspath('.')
sys.path.append(PROJECT_ROOT)
from stsm_prediction_model.error_handling import ModelError
from stsm_prediction_model.error_handling import DBError
from stsm_prediction_model.error_handling import UserRequestError
from flask import Flask, request, send_from_directory
import json
from logger import Logger
from stsm_prediction_model.stsm_model import StsmPredictionModel

app = Flask(__name__)
stsm_model = None


# TODO comments

@app.route('/stsm/algorithms/predict', methods=['POST'])
def predict():
    try:
        stsm_model.predict(request.get_json())
        logger.info('Prediction process finished successfully')
        return json.dumps({'msg': 'Prediction process was done successfully', 'success': True})
    except (DBError, ModelError, UserRequestError) as err:
        logger.error('Prediction process failed -  %s, %s' % (err.msg, err.error))
        return json.dumps({'msg': 'Prediction process failed  - %s, %s' % (err.msg, err.error), 'success': False})


@app.route('/stsm/algorithms/validate', methods=['POST', 'GET'])
def validate():
    try:
        validation_file = stsm_model.validate(request.get_json())
        logger.info('Validation process finished successfully')
        return send_from_directory(PROJECT_ROOT, validation_file, as_attachment=True)
    except (DBError, ModelError, UserRequestError) as err:
        logger.error('Validation process failed -  %s, %s' % (err.msg, err.error))
        return json.dumps({'msg': ' %s ' % err.msg, 'success': False})


def signal_handler(signal, frame):
    logger.info('Received signal - %s' % signal)
    stsm_model.disconnect()
    sys.exit(0)


if __name__ == '__main__':

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger = Logger().get_logger()
        logger.info('Starting web server...')
        stsm_model = StsmPredictionModel(logger)
        try:
            stsm_model.load_model()
        except ModelError as e:
            logger.error(e.msg)
            raise
        try:
            stsm_model.connect()
        except DBError as e:
            logger.error(e.msg)
            raise
        app.run(host='0.0.0.0', port=3100)

    except :
        logger.error('Closing prediction server')
        try:
            stsm_model.disconnect()
        except DBError as e:
            logger.error(e.msg)
