import os
import shutil
import sys

import runtime.tnt_config as tnt_config
from runtime.tnt_config import TNTConfig

def path_to_gaspi_run():
  path_to_gpi = shutil.which("gaspi_run")
  if path_to_gpi is None:
    sys.exit("[TNT_LIB] Cannot execute `gaspi_run`; make sure it is added to the current `PATH`.")
  return path_to_gpi

def path_to_gpi2_libs():
  gpi2_install_root = os.path.dirname(os.path.dirname(path_to_gaspi_run()))
  for libdir in ["lib", "lib64"]:
    gpi2_libdir = os.path.join(gpi2_install_root, libdir)
    if os.path.isdir(gpi2_libdir):
      break
  if not os.path.isdir(gpi2_libdir):
    sys.exit(f"[TNT_LIB] Cannot find `GPI-2` libraries in `{gpi2_install_root}`; "
              "make sure GPI-2 is installed using the default directory hierarchy "
              "or manually add the GPI-2 `lib` directory to `LD_LIBRARY_PATH`")
  return gpi2_libdir

def get_tnt_variables_from_args(args):
  tnt_vars = {TNTConfig.TNT_LOG_LEVEL.name : args.log_level,
              TNTConfig.TNT_LOG_ON_ALL_DEVICES.name : str(args.log_all),
              TNTConfig.TNT_OUTPUT_ON_ALL_DEVICES.name : str(args.output_all)}

  if args.fusion_threshold_kb is not None:
    tnt_vars[TNTConfig.TNT_FUSION_THRESHOLD.name] = int(args.fusion_threshold_kb) * 1024
  return tnt_vars

def get_tnt_gpus(gpus_per_node):
  return {TNTConfig.TNT_GPUS_PER_NODE.name : gpus_per_node}

def get_environment_vars_from_args(args):
  envs = {}
  for env in args.setenv:
    try:
      env_name, env_value = env.split("=")
      envs[env_name] = f"\"{env_value}\""
    except:
      raise ValueError("[LogicError] Specify environment variables as a space-separated KEY=VALUE list. "+\
                       "VALUE strings containing spaces must be enclosed in quotes.")
  return envs

def add_tnt_to_environment_paths(libraries_path):
  os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([libraries_path,
                                                   os.environ.get("LD_LIBRARY_PATH", "")])

def add_dependencies_to_environment_paths():
  os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([path_to_gpi2_libs(),
                                                   os.environ.get("LD_LIBRARY_PATH", "")])
  os.environ["PYTHONPATH"] = os.pathsep.join(sys.path +
                                             [os.environ.get("LD_LIBRARY_PATH", ""),
                                              os.environ.get("PYTHONPATH", "")])

def collect_environment_variables():
  env = {}
  for var in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']:
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
