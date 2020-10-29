import os

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
