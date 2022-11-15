import logging
logger = logging.getLogger("TNT_LIB")

import runtime.tnt_config as tnt_config
global_tnt_config = tnt_config.TarantellaConfiguration()

import pygpi
from pygpi import Allgatherv
from pygpi import Allreduce
from pygpi import Barrier
from pygpi import Broadcast
from pygpi import Group
from pygpi import ReductionOp

def get_size() -> int:
  return pygpi.get_size()

def get_rank() -> int:
  return pygpi.get_rank()

def get_group_master_rank(group: pygpi.Group) -> int:
  # use largest rank in the group as master to match partition assignment, such that ranks are assigned
  # to partition IDs in order starting from 0
  # All printing of results should happen on the last partition (that has the final outputs/metrics)
  return group.to_global_rank(group.size - 1)

def get_master_rank() -> int:
  return get_group_master_rank(pygpi.Group())

def is_master_rank():
  return get_rank() == get_master_rank()

def is_group_master_rank(group: pygpi.Group) -> bool:
  # use largest rank in the group as master to match partition assignment, such that ranks are assigned
  # to partition IDs in order starting from 0
  # All printing of results should happen on the last partition (that has the final outputs/metrics)
  return get_rank() == get_group_master_rank(group)

from tarantella import tnt_initializer
tnt_initializer.init()

from tarantella import data

from tarantella.strategy.parallel_strategy import ParallelStrategy
from tarantella.strategy.SynchCommunicator import SynchCommunicator
from tarantella.strategy.PipelineCommunicator import PipelineCommunicator

from tarantella.keras import callbacks
from tarantella.keras import models
from tarantella.keras.model import Model
from tarantella.keras.sequential import Sequential

from tarantella.collectives.TensorAllreducer import TensorAllreducer
from tarantella.collectives.TensorBroadcaster import TensorBroadcaster
from tarantella.collectives.TensorAllgatherer import TensorAllgatherer

import tarantella.optimizers as optimizers
from tarantella.optimizers.synchronous_distributed_optimizer import Optimizer

