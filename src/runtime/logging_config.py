import logging

from runtime import tf_config

def setup_logging(logger, log_level):
  logger.setLevel(log_level)
  handler = logging.StreamHandler()
  handler.setLevel(logger.level)

  formatter = logging.Formatter('[%(levelname)s][%(name)s] - %(pathname)s:%(lineno)d: %(message)s')
  handler.setFormatter(formatter)

  logger.addHandler(handler)
  tf_config.setup_logging(log_level)

