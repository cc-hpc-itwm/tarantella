import logging
import os

from runtime import tf_config

def setup_logging(logger, log_level, rank = 0, is_master_rank = True,
                  log_on_all_devices = False):
  do_logging = log_on_all_devices
  if not do_logging:
    do_logging = is_master_rank

  if log_on_all_devices:
    tnt_formatter_prefix = '[%(name)s] %(levelname)s: [rank %(rank)d] '
  else:
    tnt_formatter_prefix = '[%(name)s] %(levelname)s: '

  formatter = logging.Formatter(tnt_formatter_prefix + '%(pathname)s:%(lineno)d: %(message)s')
  logger.setLevel(log_level)
  logger.addFilter(TntLoggingFilter(rank, do_logging))

  handler = logging.StreamHandler()
  handler.setLevel(logger.level)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  tf_config.setup_logging(log_level = log_level,
                          formatter_prefix = tnt_formatter_prefix,
                          logging_filter = TntLoggingFilter(rank, do_logging))

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
