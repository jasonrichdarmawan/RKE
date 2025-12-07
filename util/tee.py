class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level, echo=None):
        self.logger = logger
        self.level = level
        self.echo = echo

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
        if self.echo is not None:
            self.echo.write(buf)

    def flush(self):
        if self.echo is not None:
            self.echo.flush()