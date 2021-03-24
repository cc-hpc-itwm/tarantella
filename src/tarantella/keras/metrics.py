import tensorflow as tf

class ZeroMetric(tf.keras.metrics.Metric):

  def __init__(self, name='zero_metric', **kwargs):
    super(ZeroMetric, self).__init__(name=name, **kwargs)
    self.zero = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    self.zero.assign(0)

  def result(self):
    return self.zero

  def reset_states(self):
    self.zero.assign(0)
