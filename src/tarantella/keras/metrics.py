import tensorflow as tf
import tarantella.utilities.tf_version as version_utils

class ZeroMetric(tf.keras.metrics.Metric):

  def __init__(self, name='zero_metric', **kwargs):
    super(ZeroMetric, self).__init__(name=name, **kwargs)
    self.zero = self.add_weight(name='tp', initializer='zeros')
    self._add_support_for_deprecated_methods()

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    self.zero.assign(0)

  def result(self):
    return self.zero

  def reset_state(self):
    self.zero.assign(0)

  def _add_support_for_deprecated_methods(self):
    if version_utils.tf_version_below_equal('2.4'):
      self.reset_states = self.reset_state
