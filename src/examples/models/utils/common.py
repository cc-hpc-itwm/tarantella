# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common util functions and classes used by both keras cifar and imagenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np
import tensorflow as tf

from models.utils import keras_utils

BASE_LEARNING_RATE = 0.1  # This matches Jing's version.
TRAIN_TOP_1 = 'training_accuracy_top_1'

class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Attributes:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, num_images):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.batches_per_epoch = num_images // batch_size # ignore the remainder
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'learning_rate'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    """Executes before step begins."""
    lr = self.schedule(self.epochs,
                       batch,
                       self.batches_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      self.prev_lr = lr
      tf.compat.v1.logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler '
          'change learning rate to %s.', self.epochs, batch, lr)


class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self, batch_size, epoch_size, warmup_epochs, boundaries,
               multipliers, compute_lr_on_cpu=True, name=None, 
               base_learning_rate = BASE_LEARNING_RATE):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
      raise ValueError('The length of boundaries must be 1 less than the '
                       'length of multipliers')

    base_lr_batch_size = 256
    num_batches_per_epoch = epoch_size // batch_size

    self.rescaled_lr = base_learning_rate * batch_size / base_lr_batch_size
    self.step_boundaries = [float(num_batches_per_epoch) * x
                            for x in boundaries]
    self.lr_values = [self.rescaled_lr * m for m in multipliers]
    self.warmup_steps = warmup_epochs * num_batches_per_epoch
    self.compute_lr_on_cpu = compute_lr_on_cpu
    self.name = name

    self.learning_rate_ops_cache = {}

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    """Compute learning rate at given step."""
    with tf.compat.v1.name_scope(self.name, 'PiecewiseConstantDecayWithWarmup',
                                 [self.rescaled_lr, self.step_boundaries,
                                  self.lr_values, self.warmup_steps,
                                  self.compute_lr_on_cpu]):
      def warmup_lr(step):
        return self.rescaled_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
      def piecewise_lr(step):
        return tf.compat.v1.train.piecewise_constant(
            step, self.step_boundaries, self.lr_values)
      return tf.cond(step < self.warmup_steps,
                     lambda: warmup_lr(step),
                     lambda: piecewise_lr(step))

  def get_config(self):
    return {
        'rescaled_lr': self.rescaled_lr,
        'step_boundaries': self.step_boundaries,
        'lr_values': self.lr_values,
        'warmup_steps': self.warmup_steps,
        'compute_lr_on_cpu': self.compute_lr_on_cpu,
        'name': self.name
    }


def build_stats(history, eval_output, callbacks):
  """Normalizes and returns dictionary of stats.

  Args:
    history: Results of the training step. Supports both categorical_accuracy
      and sparse_categorical_accuracy.
    eval_output: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    callbacks: a list of callbacks which might include a time history callback
      used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if eval_output:
    stats['accuracy_top_1'] = eval_output[1].item()
    stats['eval_loss'] = eval_output[0].item()

  if history and history.history:
    train_hist = history.history
    # Gets final loss from training.
    stats['loss'] = train_hist['loss'][-1].item()
    # Gets top_1 training accuracy.
    if 'categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = train_hist['categorical_accuracy'][-1].item()
    elif 'sparse_categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = train_hist['sparse_categorical_accuracy'][-1].item()

  if not callbacks:
    return stats
  return stats


