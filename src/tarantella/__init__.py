import logging
logger = logging.getLogger("TNT_LIB")

import runtime.tnt_config as tnt_config
global_tnt_config = tnt_config.TarantellaConfiguration()

import GPICommLib
def get_size():
  return GPICommLib.get_size()

def get_rank():
  return GPICommLib.get_rank()

def get_master_rank():
  return 0

def is_master_rank():
  return get_rank() == get_master_rank()

from tarantella import tnt_initializer
tnt_initializer.init()

from tarantella import models
from tarantella.model import Model
import tarantella.optimizers as optimizers
import tarantella.optimizers.synchronous_distributed_optimizer as distributed_optimizers
from tarantella.parallel.Barrier import Barrier
from tarantella.parallel.PipelineCommunicator import PipelineCommunicator
from tarantella.parallel.TensorAllreducer import TensorAllreducer
from tarantella.parallel.TensorBroadcaster import TensorBroadcaster
from tarantella.parallel.SynchCommunicator import SynchCommunicator
