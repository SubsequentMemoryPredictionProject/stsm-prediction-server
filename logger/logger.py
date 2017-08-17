import logging
from .sumologic_logger import SumoLogicHandler
from .http_formatter import HttpFormatter

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    def __init__(self,
                 name='app-logger',
                 level=logging.INFO,
                 log_to_console=True,
                 sumologic_collector_url=None
                 ):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(level=level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)

        if sumologic_collector_url:
            sumoLogicHandler = SumoLogicHandler()
            sumoLogicHandler.set_collector(collector_url=sumologic_collector_url)
            sumoLogicHandler.setLevel(level=level)
            formatter = HttpFormatter()
            sumoLogicHandler.setFormatter(formatter)
            logger.addHandler(sumoLogicHandler)

        self.logger = logger

    def get_logger(self):
        return self.logger
