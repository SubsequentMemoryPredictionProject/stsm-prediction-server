class ModelError(Exception):
    def __init__(self, msg, code, error):
        self.msg = msg
        self.code = code
        self.error = error


class DBError(Exception):
    def __init__(self, msg, code, error):
        self.msg = msg
        self.code = code
        self.error = error


class UserRequestError(Exception):
    def __init__(self, msg, code, error):
        self.msg = msg
        self.code = code
        self.error = error

