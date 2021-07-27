import tarantella as tnt
import tarantella.keras.layers as tnt_layers
import tarantella.keras.losses as tnt_losses

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pytest

@pytest.mark.min_tfversion('2.2')
class TestPipelineLayers:

  test_case_list = [
    {"connection_id" : 42,
     "micro_batch_size" : 2,
     "num_micro_batches" : 3,
     "data_to_be_sent" : [[27, 28, 29, 30], [7, 6, 5, 4],
                         [1, 2, 3, 4], [5, 6, 7, 8],
                         [8, 9, 10, 11], [12, 13, 14, 15]]
    },
    {"connection_id" : 0,
     "micro_batch_size" : 1,
     "num_micro_batches" : 1,
     "data_to_be_sent" : [[42]]
    },
    {"connection_id" : 13,
     "micro_batch_size" : 1,
     "num_micro_batches" : 2,
     "data_to_be_sent" : [ [[42, 3, 3], [8, 29.3, 10]],
                           [[3, 29.3, 299], [2938, 28.3, 0.1]] ]
    }
  ]

  @pytest.mark.parametrize("test_case", test_case_list)
  def test_send_recv_layers_forward(self, test_case):
    rank_0 = 0
    rank_1 = 1
    rank = tnt.get_rank()

    number_tags = 2 # mbatch_id, connection_id
    elem_type = np.dtype(np.float32)

    connection_id = test_case["connection_id"]
    micro_batch_size = test_case["micro_batch_size"]
    num_micro_batches = test_case["num_micro_batches"]
    data_to_be_sent = test_case["data_to_be_sent"]
    tensor_size = np.array(data_to_be_sent[0]).size

    tags_dataset = []
    for mbatch_id in range(num_micro_batches):
      tags_dataset = tags_dataset + micro_batch_size * [[mbatch_id, connection_id]]

    connection_table = {connection_id: ((rank_0, rank_1), tensor_size * elem_type.itemsize)}
    pipeline_comm = tnt.PipelineCommunicator(connection_table, micro_batch_size, num_micro_batches)

    if rank == rank_0:
      input_tags = keras.Input(shape=(number_tags,), name="tags", dtype = tf.int32)
      inputs = keras.Input(shape=(tensor_size,), name = 'input')
      outputs = tnt_layers.SendLayer(pipeline_communicator = pipeline_comm)(inputs, input_tags)
      model = keras.Model([inputs, input_tags], outputs)
      dataset = data_to_be_sent

    if rank == rank_1:
      input_tags = keras.Input(shape=(number_tags,), name="tags", dtype = tf.int32)
      inputs = keras.Input(shape=(tensor_size,), name = 'input')
      outputs = tnt_layers.RecvLayer(pipeline_communicator = pipeline_comm)(inputs, input_tags)
      model = keras.Model([inputs, input_tags], outputs)
      dataset = np.zeros_like(data_to_be_sent)

    def generator():
      for data, tag in zip(dataset, tags_dataset):
        yield {"input": data, "tags": tag}
    final_dataset = tf.data.Dataset.from_generator(generator,
                                                   output_types={"input": tf.float32, "tags": tf.int32})

    data_received = model.predict(final_dataset.batch(micro_batch_size))
    if rank == rank_1:
      assert np.allclose(data_received, data_to_be_sent)

  @pytest.mark.parametrize("test_case", test_case_list)
  def test_send_recv_layers_forward_and_backward(self, test_case):
    rank_0 = 0
    rank_1 = 1
    rank = tnt.get_rank()

    number_tags = 2 # mbatch_id, connection_id
    elem_type = np.dtype(np.float32)
    number_epochs = 3

    connection_id = test_case["connection_id"]
    micro_batch_size = test_case["micro_batch_size"]
    num_micro_batches = test_case["num_micro_batches"]
    data_to_be_sent = test_case["data_to_be_sent"]
    tensor_size = np.array(data_to_be_sent[0]).size

    tags_dataset = []
    for mbatch_id in range(num_micro_batches):
      tags_dataset = tags_dataset + micro_batch_size * [[mbatch_id, connection_id]]

    labels_dataset = micro_batch_size * num_micro_batches * [0.]

    connection_table = {connection_id: ((rank_0, rank_1), tensor_size * elem_type.itemsize)}
    pipeline_comm = tnt.PipelineCommunicator(connection_table, micro_batch_size, num_micro_batches)

    if rank == rank_0:
      input_tags = keras.Input(shape=(number_tags,), name="tags", dtype = tf.int32)
      input_seq = keras.Input(shape = (1,), name='input_seq')
      inputs = keras.Input(shape=(tensor_size,), name = 'input')
      outputs = tnt_layers.RemoveSeqInput()([inputs, input_seq]) # force execution of backward pass
      outputs = tnt_layers.SendLayer(pipeline_communicator = pipeline_comm)(outputs, input_tags)
      outputs = tnt_layers.IdentityLayer(name = "result")(outputs)
      model = keras.Model([inputs, input_tags, input_seq], outputs)
      loss = tnt_losses.ZeroLoss()
      dataset = data_to_be_sent

    if rank == rank_1:
      input_tags = keras.Input(shape=(number_tags,), name="tags", dtype = tf.int32)
      input_seq = keras.Input(shape = (1,), name='input_seq')
      inputs = keras.Input(shape=(tensor_size,), name = 'input')
      outputs = tnt_layers.RemoveSeqInput()([inputs, input_seq])
      outputs = tnt_layers.RecvLayer(pipeline_communicator = pipeline_comm)(outputs, input_tags)
      outputs = tnt_layers.IdentityLayer(name = "result")(outputs)
      model = keras.Model([inputs, input_tags, input_seq], outputs)
      # loss = PlusOneLoss()
      loss = tnt_losses.ZeroLoss()
      dataset = np.zeros_like(data_to_be_sent)

    def generator():
      input_seq_constant = 0
      for data, tag, label in zip(dataset, tags_dataset, labels_dataset):
        yield ({"input": data, "tags": tag, "input_seq": input_seq_constant},
               {"result": label})
    final_dataset = tf.data.Dataset.from_generator(generator, output_types=({"input": tf.float32,
                                                                             "tags": tf.int32,
                                                                             "input_seq": tf.float32},
                                                                            {"result": tf.float32}))
    final_dataset = final_dataset.batch(micro_batch_size)

    model.compile(optimizer = keras.optimizers.SGD(learning_rate=0.1), loss = loss)
    history = model.fit(final_dataset, epochs = number_epochs)
    assert len(history.history['loss']) == number_epochs
