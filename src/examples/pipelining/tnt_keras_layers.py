import tensorflow as tf
from tnt_tfops import tnt_ops

class SendLayer(tf.keras.layers.Layer):
  ''' Fwd: send output[`micro_batch_id`] to the layer on the remote rank 
      connected via `connection_id`
      Bwd: receive corresponding gradient from ibidem
  '''
  def __init__(self, pipeline_communicator):
    super().__init__()
    self.pipeline_comm_ptr = pipeline_communicator.get_raw_ptr()
  
  def build(self, input_shape):
    return super(SendLayer, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, inputs, connection_and_micro_batch_ids):
    @tf.custom_gradient
    def send_recv(x, tags):
      # tags = [micro_batch_id, connection_id] 
      micro_batch_id = tags[0][0]
      connection_id = tags[0][1]

      y = tnt_ops.send_op(x, connection_id = connection_id,
                          micro_batch_id = micro_batch_id,
                          tnt_pipeline_comm = self.pipeline_comm_ptr)
      def grad(dy):
        out = tnt_ops.recv_op(dy, connection_id = connection_id,
                              micro_batch_id = micro_batch_id,
                              tnt_pipeline_comm = self.pipeline_comm_ptr)
        return out, tf.zeros_like(tags)
      return y, grad

    return send_recv(inputs, connection_and_micro_batch_ids)


class RecvLayer(tf.keras.layers.Layer):
  ''' Fwd: receive output[`micro_batch_id`] from the layer on the remote rank 
      connected via `connection_id`
      Bwd: send corresponding gradient back to ibidem
  '''

  def __init__(self, pipeline_communicator):
    super().__init__()
    self.pipeline_comm_ptr = pipeline_communicator.get_raw_ptr()
  
  def build(self, input_shape):
    return super(RecvLayer, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape

  def call(self, inputs, connection_and_micro_batch_ids):
    @tf.custom_gradient
    def recv_send(x, tags):
      # tags = [micro_batch_id, connection_id] 
      micro_batch_id = tags[0][0]
      connection_id = tags[0][1]
      
      y = tnt_ops.recv_op(x, connection_id = connection_id,
                          micro_batch_id = micro_batch_id, 
                          tnt_pipeline_comm = self.pipeline_comm_ptr)
      def grad(dy):
        dx = tnt_ops.send_op(dy, connection_id = connection_id,
                             micro_batch_id = micro_batch_id,
                             tnt_pipeline_comm = self.pipeline_comm_ptr)
        return dx, tf.zeros_like(tags)
      return y, grad

    return recv_send(inputs, connection_and_micro_batch_ids)


class IdentityLayer(tf.keras.layers.Layer):

  def __init__(self, name='identity_layer'):
    super().__init__(name=name)
  
  def call(self, inputs):
    return inputs


class PrintLayer(tf.keras.layers.Layer):

  def __init__(self, context, name='print_layer'):
    self.context = context
    super().__init__(name=name+'_print')
  
  def call(self, inputs):
    @tf.custom_gradient
    def printing(inputs):
      tf.print(self.name, ":: fwd : rank ", self.context.rank, " ", inputs,
               "shape:", inputs.shape)
      def grad(dy):
        tf.print(self.name, ":: bwd : rank ", self.context.rank, " ", dy,
                 "shape:", dy.shape)
        return dy
      return inputs, grad
    return printing(inputs)


class RemoveSeqInput(tf.keras.layers.Layer):

  def __init__(self, name='remove_seq_input_layer'):
    super().__init__(name=name)
    self.fake_weight = self.add_weight(shape=(1),
                                       initializer='ones',
                                       trainable=True)
  
  def build(self, input_shape):
    super(RemoveSeqInput, self).build(input_shape)
    self.num_outputs = len(input_shape) - 1
    self.oshape = input_shape[:-1]

  def compute_output_shape(self, input_shape):
    return self.oshape

  def call(self, inputs):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('RemoveSeqInput must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))

    @tf.custom_gradient
    def remove_seq_input(*x_and_weight):
      weight = x_and_weight[-1]
      x = x_and_weight[:-1]
      def grad(*dy):
        if not isinstance(dy, list):
          dy = [dy]
        return dy + [tf.ones_like(x[-1])] + [tf.zeros_like(weight)]
      return x[:-1], grad

    if self.num_outputs == 1:
      output_tensors = tf.py_function(func=remove_seq_input,
                                      inp=[*inputs, self.fake_weight],
                                      Tout=tf.float32)
      output_tensors.set_shape(self.oshape[0])
    else:
      output_tensors = tf.py_function(func=remove_seq_input,
                                      inp=[*inputs, self.fake_weight],
                                      Tout=[tf.float32] * self.num_outputs)
      for i in range(self.num_outputs):
        output_tensors[i].set_shape(self.oshape[i])

    return output_tensors


class AddSeqOutput(tf.keras.layers.Layer):

  def __init__(self, micro_batch_size, name='add_seq_output_layer'):
    self.mb_size = micro_batch_size
    super().__init__(name=name)

  def build(self, input_shape):
    super(AddSeqOutput, self).build(input_shape)
    if isinstance(input_shape, list):
      self.oshape = input_shape + [tf.TensorShape([None, 1])]
    else:
      self.oshape = [input_shape] + [tf.TensorShape([None, 1])]

  def compute_output_shape(self, input_shape):
    return self.oshape
  
  def call(self, inputs):
    fake_output_shape = tf.TensorShape((self.mb_size, 1))
    fake_output = tf.zeros(fake_output_shape)
    if type(inputs) is list and len(inputs) >= 2:
      outputs = inputs + [fake_output]
    else:
      outputs = [inputs] + [fake_output]

    def add_seq_output(*inputs):
      return inputs

    output_tensors = tf.py_function(func=add_seq_output,
                                    inp=outputs,
                                    Tout=[tf.float32] * len(outputs))
    for i in range(len(output_tensors)):
      output_tensors[i].set_shape(self.oshape[i])

    return output_tensors


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
