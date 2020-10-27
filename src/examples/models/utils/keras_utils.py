
"""Helper functions for the Keras implementations of models."""
import time
import os
import sys
import copy 

import tensorflow as tf

class RuntimeProfiler(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_file_path = None, logging_freq = 10, print_freq = 30):
    """Callback for logging performance.

    Args:
      log_file_path: file to which to print log records (print to stdout if not specified).
      logging_freq: How often to record stats (number of iterations).
      print_freq: How often to print stats (number of iterations before dumping the collected stats to the log file).
    """
    super(RuntimeProfiler, self).__init__()
    self.log_file_path = log_file_path
    if self.log_file_path:
      if not os.path.dirname(self.log_file_path):
        sys.exit("Cannot create log file at " + str(log_file_path))

      log_dir = os.path.dirname(self.log_file_path)
      if not os.path.isdir(log_dir):
        print("Creating logging directory: " + log_dir)
        os.makedirs(log_dir, exist_ok = True)

      if os.path.isfile(self.log_file_path):
        print("WARNING: Log file already exists, new records will be appended to it: " + log_dir)

    self.logging_freq = logging_freq
    self.print_freq = print_freq
    self.batch_log_records = []
    self.current_epoch = 0
    self.batch_size = batch_size
    self.n_iterations = 0
    self.epoch_start_time = 0
    self.epoch_end_time = 0

  def on_train_start(self, logs=None):
    self.train_start_time = time.time()

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_epoch_begin(self, epoch, logs=None):
    self.current_epoch = epoch
    self.epoch_start_time = time.time()

  def on_train_batch_begin(self, batch, logs=None):
    self.batch_start_time = time.time()

  def on_train_batch_end(self, batch, logs=None):
    self.n_iterations = batch + 1 # batches are indexed from 0
    self.epoch_end_time = time.time()
    if batch % self.logging_freq == 0:
      iter_runtime = time.time() - self.batch_start_time
      batch_info = copy.deepcopy(logs)
      batch_info.update({'epoch' : self.current_epoch,
                         'iteration' : batch,
                         'iter_runtime_s' : iter_runtime})
      self.batch_log_records.append(batch_info)

    if batch % self.print_freq == 0:
      self.print_records(self.batch_log_records, prefix = "Iteration")
      self.batch_log_records = []

  def on_epoch_end(self, epoch, logs=None):
    epoch_runtime = self.epoch_end_time - self.epoch_start_time
    epoch_info = copy.deepcopy(logs)
    epoch_info.update({'epoch' : epoch, 
                       'size' : self.batch_size,
                       'niterations': self.n_iterations,
                       'epoch_runtime_s' : epoch_runtime})
    if len(self.batch_log_records) > 0:
      self.print_records(self.batch_log_records, prefix = "Iteration")
      self.batch_log_records = []
    self.print_records([epoch_info], prefix = "Epoch")

  def print_records(self, records, prefix = ""):
    if self.log_file_path:
      with open(self.log_file_path, "a") as f:
        for record in records:
          print(prefix + " " + str(record), file=f)
    else:
      for record in records:
        print(prefix + " " + str(record))

