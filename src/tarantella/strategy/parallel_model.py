import tarantella as tnt
from tarantella import logger

import tensorflow as tf
import abc
from abc import abstractmethod
import atexit

class ParallelModel(tf.keras.models.Model, metaclass = abc.ABCMeta):
  def __init__(self, model, group = tnt.Group()):
    super().__init__()
    self.rank = tnt.get_rank()
    self.group = group
    self.model = model
    atexit.register(self.close)

  @abstractmethod
  def close(self):
    pass

  @property
  @abstractmethod
  def metrics(self):
    pass

  @property
  @abstractmethod
  def metrics_names(self):
    pass

  @abstractmethod
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              **kwargs):
    pass

  @abstractmethod
  def evaluate(self,
               x = None,
               y = None,
               callbacks = None,
               tnt_micro_batch_size = None,
               tnt_distribute_dataset = True,
               **kwargs):
    pass

  @abstractmethod
  def fit(self,
          x = None,
          y = None,
          callbacks = None,
          validation_data = None,
          tnt_micro_batch_size = None,
          tnt_validation_micro_batch_size = None,
          tnt_distribute_dataset = True,
          tnt_distribute_validation_dataset = True,
          **kwargs):
    pass

  @abstractmethod
  def get_weights(self):
    pass

  @abstractmethod
  def load_weights(self, filepath, **kwargs):
    pass

  @abstractmethod
  def predict(self,
              x = None,
              callbacks = None,
              tnt_micro_batch_size = None,
              tnt_distribute_dataset = True,
              **kwargs):
    pass

  @abstractmethod
  def save(self, filepath, tnt_save_all_devices = False, **kwargs):
    pass

  @abstractmethod
  def save_weights(self, filepath, tnt_save_all_devices = False, **kwargs):
    pass

  @abstractmethod
  def set_weights(self, weights):
    pass


  ##############
  # Attributes #
  ##############
  @property
  def distribute_strategy(self):
    return tf.distribute.get_strategy()

  @property
  def dynamic(self):
    return self.model.dynamic

  @property
  def input_spec(self):
    return self.model.input_spec

  @property
  def layers(self):
    if hasattr(self, 'model'):
      return self.model.layers
    # condition needed for super(Model, self).__init__() to pass without error,
    # as self.model does not exist at the time of init call
    else:
      return super().layers

  @property
  def losses(self):
    return self.model.losses

  @property
  def non_trainable_weights(self):
    return self.model.non_trainable_weights

  @property
  def output(self):
    return self.model.output

  @output.setter
  def output(self, value):
    self.model.output = value

  @property
  def run_eagerly(self):
    return self.model.run_eagerly

  @property
  def state_updates(self):
    return self.model.state_updates

  @property
  def stateful(self):
    return self.model.stateful

  @property
  def trainable_weights(self):
    return self.model.trainable_weights

  @property
  def weights(self):
    return self.model.weights

  #############
  # Functions #
  #############
  def add_loss(self, losses, *args, **kwargs):
    self.model.add_loss(losses, *args, **kwargs)

  def add_metric(self, value, *args, **kwargs):
    self.model.add_metric(value, *args, **kwargs)

  def build(self, input_shape):
    return self.model.build(input_shape)

  def call(self, inputs):
    return self.model.call(inputs)

  def compute_mask(self, inputs, mask):
    return self.model.compute_mask(inputs, mask)

  def compute_output_shape(self, input_shape):
    return self.model.compute_output_shape(input_shape)

  @classmethod
  def from_config(cls, config, **kwargs):
    try:
      keras_model = tf.keras.Model.from_config(config, **kwargs)
      logger.info("Loaded model from `keras.Model`.")
    except:
      raise RuntimeError("""[tnt.Model.from_config] Cannot load
            model; provided configuration is neither a `keras.Model`
            nor a `tnt.Model`.""")
    return cls(keras_model)

  def get_config(self):
    return self.model.get_config()

  def get_layer(self, name=None, index=None):
    return self.model.get_layer(name, index)

  def reset_metrics(self):
    self.model.reset_metrics()

  def reset_states(self):
    self.model.reset_states()

  def summary(self, *args, **kwargs):
    if tnt.global_tnt_config.output_on_all_devices:
      self.model.summary(*args, **kwargs)
    else:
      if tnt.is_group_master_rank(self.group):
        self.model.summary(*args, **kwargs)

  def to_json(self, **kwargs):
    return self.model.to_json(**kwargs)

  def to_yaml(self, **kwargs):
    return self.model.to_yaml(**kwargs)
