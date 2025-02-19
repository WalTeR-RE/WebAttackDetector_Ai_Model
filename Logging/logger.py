import logging
import sys
import os

LOG_DIR = "./Logs"
os.makedirs(LOG_DIR, exist_ok=True)

def remove_logger_handlers(logger_name):
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.handlers.clear()

class CoolLogger:
    def __init__(self, name="CoolLogger", log_file="app.log", level=logging.DEBUG):
        """
        Initializes the logger with console and file handlers.

        :param name: Logger name
        :param log_file: Log file path
        :param level: Logging level (default: DEBUG)
        """
        remove_logger_handlers(name)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        log_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_colored_formatter())

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _get_colored_formatter(self):
        """ Returns a colorized log formatter """
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                "DEBUG": "\033[94m",  # Blue
                "INFO": "\033[92m",  # Green
                "WARNING": "\033[93m",  # Yellow
                "ERROR": "\033[91m",  # Red
                "CRITICAL": "\033[41m",  # Background Red
                "RESET": "\033[0m",
            }

            def format(self, record):
                log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
                reset = self.COLORS["RESET"]
                return f"{log_color}{record.levelname} | {record.getMessage()}{reset}"

        return ColoredFormatter("%(levelname)s | %(message)s")

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


logging.root.handlers.clear()
remove_logger_handlers("RepoHandlerLogger")
remove_logger_handlers("ModelLogger")

# Initialize loggers
log = CoolLogger(name="RepoHandlerLogger", log_file=f"{LOG_DIR}/repo_handler.log")
model_log = CoolLogger(name="ModelLogger", log_file=f"{LOG_DIR}/model_trainer.log")