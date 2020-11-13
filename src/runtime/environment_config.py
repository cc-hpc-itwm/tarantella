import os
import sys

import tensorflow as tf

TARANTELLA_ENV_VAR_PREFIX = "TNT_"

def get_logging_variables(log_level, log_all, output_all):
  return { "TNT_LOG_LEVEL" : str(log_level),
           "TNT_LOG_ON_ALL_DEVICES" : str(log_all),
           "TNT_OUTPUT_ON_ALL_DEVICES" : str(output_all),
          }
          
def update_environment_paths(libraries_path):
  os.environ["PYTHONPATH"]=os.pathsep.join(sys.path)

  for var_name in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
    os.environ[var_name] = os.pathsep.join([libraries_path,
                                            os.environ.get(var_name, "")])

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
    if var.startswith(TARANTELLA_ENV_VAR_PREFIX):
      env[var] = value
  return env

def gen_exports_from_dict(env_dict):
  environment = ""
  for var_name,value in env_dict.items():
    environment += "export {}={}\n".format(var_name, value)
  return environment
