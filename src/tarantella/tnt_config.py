import enum
import os
from tarantella import logger

class TNTConfig(enum.Enum):
  TNT_DEVICES_PER_NODE = 'TNT_DEVICES_PER_NODE'
  TNT_LOG_ON_ALL_DEVICES = 'TNT_LOG_ON_ALL_DEVICES'
  TNT_LOG_DIR = 'TNT_LOG_DIR'
  TNT_FUSION_THRESHOLD = 'TNT_FUSION_THRESHOLD'

class TarantellaConfigurationDefaults:
  @classmethod
  def config(self):
    default_config = { TNTConfig.TNT_DEVICES_PER_NODE : None,
                       TNTConfig.TNT_FUSION_THRESHOLD : 32 * 1024,
                       TNTConfig.TNT_LOG_ON_ALL_DEVICES : False,
                       TNTConfig.TNT_LOG_DIR : None
                     }
    return default_config

def get_configuration_from_env(filter_prefix = None):
  config = dict()
  for key in os.environ:
    if filter_prefix is None:
      config[key] = os.environ[key]
    else:
      if key.startswith(filter_prefix):
        config[key] = os.environ[key]
  return config

class TarantellaConfiguration:
  def __init__(self):
    self.tarantella_env_prefix = "TNT_"
    self.config = get_configuration_from_env(self.tarantella_env_prefix)

  def get_variable_or_default(self, variable_name):
    env_var_name = TNTConfig(variable_name).name
    value = self.config.get(env_var_name)
    if value is None:
      value = TarantellaConfigurationDefaults.config().get(variable_name)
    logger.warn("{}={}".format(env_var_name, value))
    return value

  @property
  def devices_per_node(self):
    devices_per_node_string = self.get_variable_or_default(TNTConfig.TNT_DEVICES_PER_NODE)
    if devices_per_node_string is None:
      return None
    return int(devices_per_node_string)

  @property
  def log_on_all_devices(self):
    return self.get_variable_or_default(TNTConfig.TNT_LOG_ON_ALL_DEVICES)

  @property
  def log_dir(self):
    return self.get_variable_or_default(TNTConfig.TNT_LOG_DIR)

  @property
  def fusion_threshold(self):
    return int(self.get_variable_or_default(TNTConfig.TNT_FUSION_THRESHOLD))
