import os
import sys

import runtime.tnt_config as tnt_config
from runtime.tnt_config import TNTConfig

def get_tnt_variables_from_args(args):
  tnt_vars = {TNTConfig.TNT_LOG_LEVEL.name : args.log_level,
              TNTConfig.TNT_LOG_ON_ALL_DEVICES.name : str(args.log_all),
              TNTConfig.TNT_OUTPUT_ON_ALL_DEVICES.name : str(args.output_all)}

  if args.fusion_threshold_kb is not None:
    tnt_vars[TNTConfig.TNT_FUSION_THRESHOLD.name] = int(args.fusion_threshold_kb) * 1024
  return tnt_vars

def get_tnt_gpus(gpus_per_node):
  return {TNTConfig.TNT_GPUS_PER_NODE.name : gpus_per_node}
          
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
    if var.startswith(tnt_config.TARANTELLA_ENV_VAR_PREFIX):
      env[var] = value
  return env

def gen_exports_from_dict(env_dict):
  environment = ""
  for var_name,value in env_dict.items():
    environment += "export {}={}\n".format(var_name, value)
  return environment
