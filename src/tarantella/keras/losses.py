import tensorflow as tf

class ZeroLoss(tf.keras.losses.Loss):

  def __init__(self, name='zero_loss'):
    super().__init__(name=name)

  def call(self, y_true, y_pred):
    # placeholder loss, providing constant loss and gradients,
    # to force TF to execute backward graph
    @tf.custom_gradient
    def compute_loss(x):
      def grad(dy):
        return tf.zeros_like(x)
      return tf.zeros(0), grad

    fwd_bwd = tf.py_function(func=compute_loss,
                             inp=[y_pred],
                             Tout=tf.float32)
    return fwd_bwd
