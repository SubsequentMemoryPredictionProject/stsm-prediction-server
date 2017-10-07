
class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class LoadModelError(Error):
    errorCode: 1000
    message: 'Failed loading saved model'
