import tensorflow as tf

class P2PLayer(tf.keras.layers.Layer):
  def __init__(self, pipeline_communicator):
    super().__init__()
    self.pipeline_comm = pipeline_communicator

  def build(self, input_shape):
    return super().build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    return super().get_config()

class SendLayer(P2PLayer):
  ''' Fwd: send output[`micro_batch_id`] to the layer on the remote rank
      connected via `connection_id`
      Bwd: receive corresponding gradient from ibidem
  '''
  def call(self, inputs, connection_and_micro_batch_ids):
    @tf.custom_gradient
    def send_recv(x, tags):
      # tags = [micro_batch_id, connection_id]
      micro_batch_id = tags[0][0]
      connection_id = tags[0][1]

      y = self.pipeline_comm.send(x, connection_id = connection_id,
                                  micro_batch_id = micro_batch_id)
      def grad(dy):
        out = self.pipeline_comm.recv(dy, connection_id = connection_id,
                                      micro_batch_id = micro_batch_id)
        return out, tf.zeros_like(tags)
      return y, grad

    return send_recv(inputs, connection_and_micro_batch_ids)


class RecvLayer(P2PLayer):
  ''' Fwd: receive output[`micro_batch_id`] from the layer on the remote rank
      connected via `connection_id`
      Bwd: send corresponding gradient back to ibidem
  '''
  def call(self, inputs, connection_and_micro_batch_ids):
    @tf.custom_gradient
    def recv_send(x, tags):
      # tags = [micro_batch_id, connection_id]
      micro_batch_id = tags[0][0]
      connection_id = tags[0][1]

      y = self.pipeline_comm.recv(x, connection_id = connection_id,
                                  micro_batch_id = micro_batch_id)
      def grad(dy):
        dx = self.pipeline_comm.send(dy, connection_id = connection_id,
                                     micro_batch_id = micro_batch_id)
        return dx, tf.zeros_like(tags)
      return y, grad

    return recv_send(inputs, connection_and_micro_batch_ids)

class SynchSendLayer(P2PLayer):
  ''' Fwd: send output[`micro_batch_id`] to the layer on the remote rank
      connected via `connection_id` and wait for acknowledgement from receiver
      Bwd: receive corresponding gradient from ibidem
  '''
  def call(self, inputs, connection_and_micro_batch_ids):
    @tf.custom_gradient
    def send_recv(x, tags):
      # tags = [micro_batch_id, connection_id]
      micro_batch_id = tags[0][0]
      connection_id = tags[0][1]

      y = self.pipeline_comm.send_with_acknowledgement(x, connection_id = connection_id,
                                                       micro_batch_id = micro_batch_id)
      def grad(dy):
        out = self.pipeline_comm.recv(dy, connection_id = connection_id,
                                      micro_batch_id = micro_batch_id)
        return out, tf.zeros_like(tags)
      return y, grad

    return send_recv(inputs, connection_and_micro_batch_ids)


class SynchRecvLayer(P2PLayer):
  ''' Fwd: receive output[`micro_batch_id`] from the layer on the remote rank
      connected via `connection_id` and send receipt acknowledgement to sender
      Bwd: send corresponding gradient back to ibidem
  '''
  def call(self, inputs, connection_and_micro_batch_ids):
    @tf.custom_gradient
    def recv_send(x, tags):
      # tags = [micro_batch_id, connection_id]
      micro_batch_id = tags[0][0]
      connection_id = tags[0][1]

      y = self.pipeline_comm.recv_with_acknowledgement(x, connection_id = connection_id,
                                                       micro_batch_id = micro_batch_id)
      def grad(dy):
        dx = self.pipeline_comm.send(dy, connection_id = connection_id,
                                     micro_batch_id = micro_batch_id)
        return dx, tf.zeros_like(tags)
      return y, grad

    return recv_send(inputs, connection_and_micro_batch_ids)


class IdentityLayer(tf.keras.layers.Layer):

  def __init__(self, name='identity_layer'):
    super().__init__(name=name)

  def call(self, inputs):
    return inputs


class PrintLayer(tf.keras.layers.Layer):

  def __init__(self, rank, name='print_layer'):
    super().__init__(name=name)
    self.rank = rank

  def call(self, inputs):
    @tf.custom_gradient
    def printing(inputs):
      tf.print(self.name, ":: fwd : rank ", self.rank, " ", inputs,
               "shape:", inputs.shape)
      def grad(dy):
        tf.print(self.name, ":: bwd : rank ", self.rank, " ", dy,
                 "shape:", dy.shape)
        return dy
      return inputs, grad
    return printing(inputs)


class RemoveSeqInput(tf.keras.layers.Layer):

  def __init__(self, name='remove_seq_input_layer'):
    super().__init__(name=name)
    # defines a fake weight (and corresponding custom gradient) in order to
    # force TF to execute the backward pass
    self.fake_weight = self.add_weight(shape=(1),
                                       initializer='ones',
                                       trainable=True)

  def build(self, input_shape):
    super().build(input_shape)
    self.num_outputs = len(input_shape) - 1
    self.oshape = input_shape[:-1]

  def compute_output_shape(self, input_shape):
    return self.oshape

  def get_config(self):
    return super().get_config()

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
    super().build(input_shape)
    if isinstance(input_shape, list):
      self.oshape = input_shape + [tf.TensorShape([None, 1])]
    else:
      self.oshape = [input_shape] + [tf.TensorShape([None, 1])]

  def get_config(self):
    return super().get_config()

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
