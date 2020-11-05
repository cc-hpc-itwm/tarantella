import os
import logging

from runtime import logger 
import tensorflow as tf

def get_logging_variables(log_all, user_log_dir):
  return { "TNT_LOG_LEVEL" : logging.getLevelName(logger.level),
           "TNT_LOG_ON_ALL_DEVICES" : log_all,
           "TNT_LOG_DIR" : user_log_dir
          }
          
def collect_environment_variables():
  env = {}
  for var in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH']:
    if var in os.environ:
      env[var] = os.environ[var]
  return env

def collect_tensorflow_variables():
  env = {}
  for var, value in os.environ.items():
    if var.lower().startswith("tf_"):
      env[var] = value
  return env

def collect_tarantella_variables():
  env = {}
  for var, value in os.environ.items():
    if var.lower().startswith("tnt_"):
      env[var] = value
  return env

def gen_exports_from_dict(env_dict):
  environment = ""
  for var_name,value in env_dict.items():
    environment += "export %s=%s\n" % (var_name, value)
  return environment
