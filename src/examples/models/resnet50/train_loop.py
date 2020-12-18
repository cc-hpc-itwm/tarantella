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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import tensorflow as tf

from models.resnet50 import imagenet_preprocessing
from models.resnet50 import resnet_model
from models.utils import common

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_dir", help="location of the ImageNet dataset")
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--train_epochs", type=int, default=90)
  parser.add_argument("--profile_dir", help="directory for profiles")
  parser.add_argument("--without-datapar",
                      action='store_true',
                      default = False)
  parser.add_argument("--shuffle-seed",
                      type = int,
                      default = 42)
  parser.add_argument("--val-freq",
                      type=int,
                      default = 1)
  parser.add_argument("--data-format",
                      help = "Reshape data into either 'channels_last' or 'channels_first' format",
                      default = "channels_last")
  parser.add_argument("--profile-runtimes", action='store_true',
                      default = False)
  parser.add_argument("--logging-freq", help="how often (in number of iterations) to record the runtimes per iteration",
                      type=int, default = 10)
  parser.add_argument("--print-freq", help="how often (in number of iterations) to print the recorded iteration runtimes",
                      type=int, default = 30)
  
  args = parser.parse_args()
  if args.data_dir == None or not os.path.isdir(args.data_dir):
    sys.exit("ERROR: Cannot find images directory %s" % args.data_dir)
  return args  

BASE_LEARNING_RATE = 0.1  # This matches Jing's version.
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]
MIN_BATCH_SIZE = 64 # min micro-batch size used for one rank

def learning_rate_schedule(current_epoch,
                           current_batch,
                           batches_per_epoch,
                           batch_size):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    batches_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  initial_lr = BASE_LEARNING_RATE * batch_size / MIN_BATCH_SIZE
  epoch = current_epoch + float(current_batch) / int(batches_per_epoch)
  warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
  if epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return initial_lr * warmup_lr_multiplier * epoch / warmup_end_epoch
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_lr * mult
    else:
      break
  return learning_rate

def load_data(data_dir, batch_size, rank, comm_size,
              shuffle_seed = None):

  micro_batch_size = batch_size // comm_size
  train_input_dataset = imagenet_preprocessing.input_fn(
      dataset_type='train',
      data_dir=data_dir,
      micro_batch_size=micro_batch_size,
      parse_record_fn=imagenet_preprocessing.parse_record,
      dtype=tf.float32,
      drop_remainder=True,
      comm_size=comm_size,
      rank=rank,
      shuffle_seed=shuffle_seed
  )
  val_input_dataset = imagenet_preprocessing.input_fn(
      dataset_type='validation',
      data_dir=data_dir,
      micro_batch_size=micro_batch_size,
      parse_record_fn=imagenet_preprocessing.parse_record,
      dtype=tf.float32,
      drop_remainder=True
  )
  return {"train" : train_input_dataset,
          "validation" : val_input_dataset }


def run(model, optimizer,
        batch_size, 
        train_epochs, 
        data_dir,
        rank = 0, comm_size = 1, 
        custom_callbacks = None,
        val_freq = 1,
        shuffle_seed = None,
        tarantella_enabled = False):
  """Run ResNet CIFAR10 training and eval loop using native Keras APIs.

  Args:

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  micro_batch_size = batch_size // comm_size
  datasets = load_data(data_dir, batch_size, rank, comm_size, shuffle_seed)

  train_steps = imagenet_preprocessing.NUM_IMAGES['train'] // batch_size
  num_val_steps = imagenet_preprocessing.NUM_IMAGES['validation'] // micro_batch_size

  callbacks = [] 
  if custom_callbacks:
    callbacks += custom_callbacks
  callbacks.append(common.LearningRateBatchScheduler(
                            learning_rate_schedule,
                            batch_size=batch_size,
                            num_images=imagenet_preprocessing.NUM_IMAGES['train']))
  
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer,
                metrics=(['sparse_categorical_accuracy']))

  kwargs = {}
  if tarantella_enabled:
    kwargs = {'tnt_distribute_dataset': False,
              'tnt_distribute_validation_dataset': False}

  history = model.fit(datasets['train'],
                      epochs=train_epochs,
                      steps_per_epoch=train_steps,
                      callbacks=callbacks,
                      validation_steps=num_val_steps,
                      validation_data=datasets['validation'],
                      validation_freq=val_freq,
                      verbose=2,
                      **kwargs)

  kwargs = {}
  if tarantella_enabled:
    kwargs = {'tnt_distribute_dataset': False}

  stats = {}
  eval_output = model.evaluate(datasets['validation'],
                                steps=num_val_steps,
                                verbose=2,
                                **kwargs)
  stats = common.build_stats(history, eval_output, callbacks)
  return stats

