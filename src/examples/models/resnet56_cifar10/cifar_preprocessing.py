# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Provides utilities to Cifar-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import tensorflow as tf

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = HEIGHT * WIDTH * NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1

NUM_IMAGES = {
    'train': 45000,
    'validation': 5000,
    'test': 10000,
}
_NUM_DATA_FILES = 5
NUM_CLASSES = 10


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  This method converts the label to one hot to fit the loss function.

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: Data type to use for input images.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.io.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [NUM_CHANNELS, HEIGHT, WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(a=depth_major, perm=[1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training)
  image = tf.cast(image, dtype)

  # TODO(haoyuzhang,hongkuny,tobyboyd): Remove or replace the use of V1 API
  label = tf.compat.v1.sparse_to_dense(label, (NUM_CLASSES,), 1)
  return image, label


def preprocess_image(image, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(
        image, HEIGHT + 8, WIDTH + 8)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  
  if tf.keras.backend.image_data_format() == "channels_first":
    image = tf.transpose(a=image, perm=[2, 0, 1])
  return image


def get_filenames(dataset_type, data_dir):
  """Returns a list of filenames.
   
   Args:
   dataset_type: String defining the purpose of the dataset (train, validation, or test)
   """
  assert tf.io.gfile.exists(data_dir), (
      'Run cifar10_download_and_extract.py first to download and extract the '
      'CIFAR-10 data.')

  if dataset_type in ["train", "validation"]:
    return [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    return [os.path.join(data_dir, 'test_batch.bin')]

def process_record_dataset(dataset,
                           is_training,
                           micro_batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           dtype=tf.float32,
                           drop_remainder=False,
                           comm_size = 1,
                           rank = 0,
                           shuffle_seed = None):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    micro_batch_size: The number of samples per micro batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    dtype: Data type to use for images/features.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  
  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=micro_batch_size)
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer,
                              seed=shuffle_seed, 
                              reshuffle_each_iteration=False)
    # Repeats the dataset indefinitely
    dataset = dataset.repeat()
    # Assign each process only the data samples with index % rank == 0
    dataset = dataset.shard(num_shards=comm_size, index = rank)

  # Parses the raw records into images and labels.
  dataset = dataset.map(
      lambda value: parse_record_fn(value, is_training, dtype),
      num_parallel_calls=4)
      #num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(micro_batch_size, drop_remainder=drop_remainder)
  

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.data.experimental.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=4)
  return dataset

def input_fn(dataset_type,
             data_dir,
             micro_batch_size,
             dtype=tf.float32,
             parse_record_fn=parse_record,
             drop_remainder=False,
             comm_size = 1,
             rank = 0,
             shuffle_seed = None):
  """Input function which provides batches for train or eval.

  Args:
    dataset_type: String defining the purpose of the dataset (train, validation, or test)
    data_dir: The directory containing the input data.
    micro_batch_size: The number of samples per micro-batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    parse_record_fn: Function to use for parsing the records.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(dataset_type, data_dir)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  is_training = False
  # reserve the first samples for the validation dataset
  if dataset_type == "train":
    dataset = dataset.skip(NUM_IMAGES['validation'])
    is_training = True
  elif dataset_type == "validation":
    dataset = dataset.take(NUM_IMAGES['validation'])

  return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      micro_batch_size=micro_batch_size,
      shuffle_buffer=NUM_IMAGES['train'],
      parse_record_fn=parse_record_fn,
      dtype=dtype,
      drop_remainder=drop_remainder,
      comm_size = comm_size,
      rank = rank,
      shuffle_seed = shuffle_seed
  )
