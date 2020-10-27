import tensorflow as tf
import pathlib
import os

tnt_ops = tf.load_op_library(os.path.join(pathlib.Path(__file__).parent.absolute(),
                                          'libtnt-tfops.so'))

