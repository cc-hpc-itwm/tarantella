import logging
import os

from runtime import tf_config

def setup_logging(logger, log_level, rank = 0, is_master_rank = True,
                  log_on_all_devices = False):
  do_logging = log_on_all_devices
  if not do_logging:
    do_logging = is_master_rank

  if log_on_all_devices:
    tnt_formatter_prefix = '[%(name)s]%(levelname)7s: [rank%(rank)3d] '
  else:
    tnt_formatter_prefix = '[%(name)s]%(levelname)7s: '
  formatter = TntPathFormatter(tnt_formatter_prefix + '%(pathname)s:%(lineno)3d: %(message)s')

  logger.setLevel(log_level)
  logger.addFilter(TntLoggingFilter(rank, do_logging))

  handler = logging.StreamHandler()
  handler.setLevel(logger.level)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  tf_config.setup_logging(formatter_type = TntPathFormatter,
                          formatter_prefix = tnt_formatter_prefix,
                          logging_filter = TntLoggingFilter(rank, do_logging))


class TntPathFormatter(logging.Formatter):
  def format(self, record):
    PATH_LEN = 40
    if len(record.pathname) > PATH_LEN:
      record.pathname = f"~{record.pathname[-(PATH_LEN):]}"
    return super().format(record)

class TntLoggingFilter(logging.Filter):
  def __init__(self, rank, do_logging):
    super().__init__()
    self.rank = rank
    self.do_logging = do_logging

  def filter(self, record):
    record.rank = self.rank
    if not self.do_logging:
      return False
    return True
