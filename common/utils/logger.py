import os
import sys
import time

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