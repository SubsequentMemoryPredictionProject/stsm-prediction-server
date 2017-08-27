import logging

class Singleton(type):
    _instances = {}
    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(Singleton, self).__call__(*args, **kwargs)
        return self._instances[self]

class Logger(metaclass=Singleton):
    def __init__(self,
                 name='stsm-prediction-server-logger',
                 level=logging.INFO,
                 log_to_console=True,
                 handler=logging.FileHandler('stsm-prediction-server.log'),
    ):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        self.logger = logger

    def get_logger(self):
        return self.logger
