from models import mnist_models as mnist
import utilities as util
import tarantella as tnt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pytest

class TestsModelAPI:
  # Attributes
  def test_distribute_strategy(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.distribute_strategy == tf.distribute.get_strategy()

  def test_dynamic(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.dynamic == False

  @pytest.mark.max_tfversion('2.3')
  def test_input_spec(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.input_spec == None

  @pytest.mark.min_tfversion('2.4')
  def test_input_spec(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.input_spec[0].shape == (None, 28, 28, 1)

  def test_layers(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert len(tnt_model.layers) == 8
    assert tnt_model.layers[1].name == 'conv1'
    
  def test_losses(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.losses == []
    tnt_model.add_loss(tf.abs(tnt_model.output))
    assert len(tnt_model.losses) == 1

  def test_metrics(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.metrics == []

  @pytest.mark.max_tfversion('2.1')
  def test_metrics_names(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.metrics_names == ['loss']

  @pytest.mark.min_tfversion('2.2')
  def test_metrics_names(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.metrics_names == []

  @pytest.mark.min_tfversion('2.2')
  def test_metrics_names_after_fit(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    tnt_model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss="sparse_categorical_crossentropy",
                      metrics=["sparse_categorical_accuracy"])
    train_dataset, _, _ = util.load_dataset(mnist.load_mnist_dataset,
                                            train_size = 24,
                                            train_batch_size = 24)
    tnt_model.fit(train_dataset)
    assert tnt_model.metrics_names == ["loss", "sparse_categorical_accuracy"]

  def test_non_trainable_weights(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.non_trainable_weights == []

  def test_output(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.output.shape[0] == None
    assert tnt_model.output.shape[1] == 10

  def test_run_eagerly(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.run_eagerly == False

  def test_state_updates(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.state_updates == []

  def test_stateful(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.stateful == False

  def test_trainable_weights(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert len(tnt_model.trainable_weights) == 8 # 2 convs, 2 dense + biases

  def test_weights(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert len(tnt_model.weights) == 8 # 2 convs, 2 dense + biases

  # Functions
  def test_add_loss(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.losses == []
    tnt_model.add_loss(tf.reduce_mean(tnt_model.output))
    assert len(tnt_model.losses) == 1

  @pytest.mark.max_tfversion('2.1')
  def test_add_metric(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.metrics == ['loss']
    tnt_model.add_metric(tnt_model.output, aggregation='mean', name='metric_name')
    assert len(tnt_model.metrics) == 2
    assert tnt_model.metrics_names == ['loss', 'metric_name']

  @pytest.mark.min_tfversion('2.2')
  @pytest.mark.max_tfversion('2.2')
  def test_add_metric(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.metrics == []
    tnt_model.add_metric(tnt_model.output, aggregation='mean', name='metric_name')  # deprecated after Tf2.2
    assert len(tnt_model.metrics) == 1
    assert tnt_model.metrics_names == ['metric_name']

  def test_compute_mask(self):
    pass

  def test_compute_output_shape(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    output_shape = tnt_model.compute_output_shape((None, 28, 28, 1))
    assert output_shape[0] == None
    assert output_shape[1] == 10

  def test_get_layer(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    assert tnt_model.get_layer(index = 0) == tnt_model.layers[0]
    assert tnt_model.get_layer(index = 0).name == 'input'
    assert tnt_model.get_layer(index = 6) == tnt_model.layers[6]

  def test_reset_metrics(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    tnt_model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss="sparse_categorical_crossentropy",
                      metrics=["sparse_categorical_accuracy"])
    train_dataset, _, _ = util.load_dataset(mnist.load_mnist_dataset,
                                            train_size = 60,
                                            train_batch_size = 60)
    tnt_model.fit(train_dataset)
    assert all(float(m.result()) != 0 for m in tnt_model.metrics)

    tnt_model.reset_metrics()
    assert all(float(m.result()) == 0 for m in tnt_model.metrics)

  def test_reset_states(self):
    tnt_model = tnt.Model(mnist.lenet5_model_generator())
    try:
        tnt_model.reset_states()
    except Exception as exc:
        assert False, f"`tnt_model.reset_states()` raised an exception {exc}"
  
  @pytest.mark.parametrize("optimizer_name, optimizer_type", [
                                                              ("sgd", tf.keras.optimizers.SGD),
                                                              ("rmsprop", tf.keras.optimizers.RMSprop)
                                                             ])
  def test_optimizer_with_name(self, optimizer_name, optimizer_type):
    tnt_model = tnt.Model(mnist.lenet5_model_generator(),
                          parallel_strategy = tnt.ParallelStrategy.DATA)
    tnt_model.compile(optimizer=optimizer_name,
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    tnt_optimizer = tnt_model.dist_optimizer
    assert isinstance(tnt_optimizer, optimizer_type)
