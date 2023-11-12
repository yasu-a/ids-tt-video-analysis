import logging
import re

_header_width = 55


class CustomFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        m = re.match(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3}\s\[[^]]*?\]', s)
        split_index = m.end()
        global _header_width
        _header_width = min(100, max(_header_width, split_index + 1))
        return s[:split_index].ljust(_header_width) + s[split_index:]


def create_logger(name) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    syslog = logging.StreamHandler()
    formatter = CustomFormatter('%(asctime)s [%(levelname)s:%(name)s] %(message)s')
    syslog.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(syslog)
    return logger
