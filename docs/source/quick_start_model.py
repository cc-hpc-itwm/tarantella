import tensorflow as tf
from tensorflow import keras
import tarantella as tnt

# Skip function implementations for brevity
[...]

# Initialize Tarantella (before doing anything else)
tnt.init()
args = parse_args()
              
# Create Tarantella model
model = tnt.Model(lenet5_model_generator())

# Compile Tarantella model (as with Keras)
model.compile(optimizer = keras.optimizers.SGD(learning_rate=args.learning_rate),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = [keras.metrics.SparseCategoricalAccuracy()])

# Load MNIST dataset (as with Keras)
shuffle_seed = 42
(x_train, y_train), (x_val, y_val), (x_test, y_test) = \
      mnist_as_np_arrays(args.train_size, args.val_size, args.test_size)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(len(x_train), shuffle_seed)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(args.batch_size)

# Train Tarantella model (as with Keras)
model.fit(train_dataset,
          epochs = args.number_epochs,
          verbose = 1)

# Evaluate Tarantella model (as with Keras)
model.evaluate(test_dataset, verbose = 1)
