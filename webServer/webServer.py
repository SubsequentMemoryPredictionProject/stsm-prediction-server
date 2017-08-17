import os
from flask import Flask, jsonify, request
import json

app = Flask(__name__)

@app.route('/test')
def test():
    return json.dumps({'ok': True})

if __name__ == '__main__':
    print('Starting web server...')
    app.run(host='0.0.0.0', port=3100)
