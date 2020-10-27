from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime 

from models.resnet50  import train_loop
from models.resnet50 import imagenet_preprocessing
from models.resnet50 import resnet_model
from models.utils import keras_utils as utils

import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

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
    tarantella.init(args.ngpus_per_node)
    have_datapar = True
  
  rank = 0
  comm_size = 1
  if have_datapar:
    rank = tarantella.get_rank()
    comm_size = tarantella.get_size()
    
  callbacks = []
  if args.profile_dir:
    # Start the training w/ logging
    log_dir = os.path.join(args.profile_dir, "gpi-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/rank" + str(rank))
    os.makedirs(log_dir, exist_ok = True)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0))
    
  if rank == 0 and args.profile_runtimes:
    callbacks += [utils.RuntimeProfiler(batch_size = args.batch_size,
                                        logging_freq = args.logging_freq,
                                        print_freq = args.print_freq) ]
    
  model = resnet_model.resnet50(
        num_classes=imagenet_preprocessing.NUM_CLASSES)
  if have_datapar:
    model = tarantella.model.TarantellaModel(model,
            _fusion_threshold_bytes = args.tensor_fusion_threshold)

  optimizer = gradient_descent_v2.SGD(lr=train_loop.BASE_LEARNING_RATE, momentum=0.9)

  train_loop.run(model, optimizer,
      args.batch_size, 
      args.train_epochs, 
      args.data_dir,
      rank, comm_size, 
      callbacks, 
      args.val_freq, 
      args.shuffle_seed)


