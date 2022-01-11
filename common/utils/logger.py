import os
import sys
import time
import logging

class TextLogger(object):
    """Writes stream output to external text file.
    Args:
        filename (str): the file to write stream output
        stream: the stream to read from. Default: sys.stdout
    """
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.terminal.close()
        self.log.close()

class CompleteLogger:
    """
    A useful logger that
    - writes outputs to files and displays them on the console at the same time.
    - manages the directory of checkpoints and debugging images.
    Reference: https://github.com/thuml/Transfer-Learning-Library/blob/master/common/utils/logger.py
    Args:
        root (str): the root directory of logger
        phase (str): the phase of training.
    """

    def __init__(self, root, phase='train'):
        self.root = root
        self.checkpoint_directory = os.path.join(self.root, "checkpoints")
        self.epoch = 0
        self.phase = phase
        os.makedirs(self.root, exist_ok=True)

        # redirect std out
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        log_filename = os.path.join(self.root, "{}-{}.txt".format(phase, now))
        if os.path.exists(log_filename):
            os.remove(log_filename)
        self.logger = TextLogger(log_filename)
        sys.stdout = self.logger
        sys.stderr = self.logger

    def set_epoch(self, epoch):
        """Set the epoch number. Please use it during training."""
        self.epoch = epoch

    def close(self):
        self.logger.close()

def get_logging_logger(name, logfile, level = logging.DEBUG, save=True):
    """
    Create a logger provided by a standard library module
    Args:
        name (str): the root directory of logger
        logfile (str): the phase of training.
        level : Sets the threshold for this logger to `level`. Logging messages which are less severe than level will be ignored, DEBUG < INFO < WARNING < ERROR < CRITICAL
    Examples::

        >>> self.logger = get_logger(model_name,os.path.join(self.model_save_folder,'logging.txt'))
        >>> self.logger.info("train_loss: 0.126655489")
        2021-12-23 17:09:44,952 - model_name - INFO - train_loss: 0.126655489
    """
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler and set level to debug
    if save:
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(level)
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        fh.close()
    ch.close()

    return logger