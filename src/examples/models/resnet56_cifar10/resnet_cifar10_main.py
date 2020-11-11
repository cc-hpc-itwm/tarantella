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

import os
import datetime

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

from models.resnet56_cifar10  import train_loop
from models.resnet56_cifar10  import cifar_preprocessing
from models.resnet56_cifar10 import resnet_cifar_model

args = train_loop.parse_args()

# Global settings
if args.data_format == "channels_first":
  print("Setting data format to 'channels_first'.")
  tf.keras.backend.set_image_data_format("channels_first")

if not args.without_datapar:
  import tarantella

if __name__ == '__main__':

  have_datapar = False
  if not args.without_datapar:
    tarantella.init()
    have_datapar = True

  rank = 0
  comm_size = 1
  if have_datapar:
    rank = tarantella.get_rank()
    comm_size = tarantella.get_size()

  callbacks = []
  if args.profile_dir and have_datapar:
    # Start the training w/ logging
    log_dir = os.path.join(args.profile_dir, "gpi-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/rank" + str(rank))
    os.makedirs(log_dir, exist_ok = True)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0))

  model = resnet_cifar_model.resnet56(
        classes=cifar_preprocessing.NUM_CLASSES)
  if have_datapar:
    model = tarantella.Model(model)

  optimizer = gradient_descent_v2.SGD(lr=train_loop.BASE_LEARNING_RATE, momentum=0.9)
  train_loop.run(model, optimizer,
      args.batch_size, args.train_epochs,
      args.data_dir,
      rank, comm_size,
      callbacks,
      args.val_freq,
      args.shuffle_seed)
