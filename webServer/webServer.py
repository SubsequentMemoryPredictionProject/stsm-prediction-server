import os
from flask import Flask, jsonify, request
import json
from logger import Logger


app = Flask(__name__)

@app.route('/test')
def test():
    return json.dumps({'ok': True})

if __name__ == '__main__':
    # logger = Logger().get_logger()
    # logger.info('Starting web server...')
    print('Starting web server...')
    app.run(host='0.0.0.0', port=3100)
