import platform
import os

import runtime.tf_config as tf_config
from runtime import logger

def generate_nodes_list(hostfile = None):
  if hostfile is None:
    hostname = platform.node()
    logger.debug("No `hostfile` provided. Using only the current node `{}`".format(hostname))
    return [hostname]

  if not os.path.isfile(hostfile):
    raise ValueError("Incorrect `hostfile` provided with path `{}`".format(hostfile))

  nodes_list = []
  try:
    with open(hostfile, 'r') as f:
      nodes_list = f.readlines()
  except:
    raise ValueError("Cannot read from `hostfile` with path `{}`".format(hostfile))

  if len(nodes_list) == 0:
    raise ValueError("Empty `hostfile` with path `{}`".format(hostfile))
  
  unique_nodes = [node.strip() for node in set(nodes_list)]
  if len(nodes_list) != len(set(nodes_list)):
    logger.debug("The `hostfile` does not contain only unique hostnames; removing duplicates.")
  return unique_nodes


def generate_num_gpus_per_node(npernode = None):
  num_physical_gpus = len(tf_config.get_available_gpus())
  logger.debug("Num GPUs Available: {}".format(num_physical_gpus))

  if npernode is None:  # use as many GPUs as possible
    num_devices = num_physical_gpus

  else: # the user requested a specific number of devices
    if num_physical_gpus < npernode:
      logger.debug("Not enough GPUs for the requested {} devices per node".format(npernode))
      num_devices = 0
    else:
      num_devices = npernode
  return num_devices

def generate_num_devices_per_node(npernode = None, use_gpus = True):
  num_gpus = 0
  if use_gpus:
    num_gpus = generate_num_gpus_per_node(npernode)

  num_cpus = 0
  if num_gpus == 0:
    if npernode is None:  # use one rank per node
      num_cpus = 1
    else:
      num_cpus = npernode

  if use_gpus and num_gpus <= 0:
    logger.warn("Cannot find {0} available GPUs per node as \
requested; using {0} ranks on CPUs instead".format(num_cpus))

  return num_gpus, num_cpus