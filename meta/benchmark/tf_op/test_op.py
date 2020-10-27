import tensorflow as tf
import numpy as np
import time

my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
#tf.debugging.set_log_device_placement(True)


time2_module = tf.load_op_library('./time2.so')
Time2 = time2_module.time2
SIZE = 100000
Epoch = 10
RANGE = 10

with tf.device('/CPU:0'):
    left = RANGE*tf.random.normal([SIZE], 0, 1, tf.float32, seed=1) 
    right = RANGE*tf.random.normal([SIZE], 0, 1, tf.float32, seed=2) 


def py_func(a,b):
    return tf.math.add(a,tf.math.scalar_mul(2.,b))

@tf.function
def py_op(left,right):
    return tf.py_function(func=py_func, inp=[left,right], Tout=tf.float32)

@tf.function
def tf_op(left,right):
    return Time2(left,right)

print(SIZE)
print(Epoch)
with tf.device('/CPU:0'):
    start = time.time()
    for i in range(Epoch):
        result = tf_op(left,right)
    end = time.time()
    print("tf_op cpu with @tf.function take time:")
    print(end-start)

with tf.device('/CPU:0'):
    start = time.time()
    for i in range(Epoch):
        result = py_op(left,right)
    end = time.time()
    print("py_func cpu with @tf.function take time:")
    print(end-start)

