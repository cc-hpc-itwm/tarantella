
from enum import Enum


class TF_verbose(Enum):
  SILENT = 0
  ALL = 1
  LESS = 2

def _set_model_optimizer(model, optimizer):
  if hasattr(model, '_get_optimizer'):
    # wrap optimizer in an internal `keras` data structure
    model.optimizer = model._get_optimizer(optimizer)
  elif hasattr(model, '_set_optimizer'):
    #for Sequential model with TF 2.0/2.1
    model._set_optimizer(optimizer)
  else:
    raise RuntimeError(
    "[tnt.keras.utilities._set_model_optimizer] Cannot set optimizer for the provided `keras.Model`.")
