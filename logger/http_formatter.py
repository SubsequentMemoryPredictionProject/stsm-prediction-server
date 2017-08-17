from logging import Formatter
import datetime
import json

class HttpFormatter(Formatter):
    def __init__(self):
        super(HttpFormatter, self).__init__()

    def format(self, record):
        data = {
            'message': record.msg,
            'timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
        }

        return json.dumps(data)