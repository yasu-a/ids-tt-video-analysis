import re
# noinspection PyUnresolvedReferences
from logging import CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET
from logging import getLogger, StreamHandler, Formatter, LogRecord, Logger

__all__ = (
    'Logger',
    'create_logger',
    'set_log_level',
    'CRITICAL',
    'FATAL',
    'ERROR',
    'WARNING',
    'WARN',
    'INFO',
    'DEBUG',
    'NOTSET'
)

_header_width = 55


class CustomFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        s = super().format(record)
        m = re.match(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s\[[^]]*?\]', s)
        split_index = m.end()
        global _header_width
        _header_width = min(100, max(_header_width, split_index + 1))
        header, rest = s[:split_index].ljust(_header_width), s[split_index:]
        # TODO: fix bug: first line starts with one more white-space at the beginning of rest
        rest = rest[1:]
        log_lines = rest.split('\n')
        return '\n'.join(header + line for line in log_lines)


_loggers = []
_default_log_level = INFO


def create_logger(name) -> Logger:
    logger = getLogger(name)
    logger.propagate = False
    syslog = StreamHandler()
    formatter = CustomFormatter('%(asctime)s [%(levelname)s:%(name)s] %(message)s')
    syslog.setFormatter(formatter)
    logger.setLevel(_default_log_level)
    logger.addHandler(syslog)

    global _loggers
    _loggers.append(logger)

    return logger


def set_log_level(level):
    for logger in _loggers:
        logger.setLevel(level)

    global _default_log_level
    _default_log_level = level
