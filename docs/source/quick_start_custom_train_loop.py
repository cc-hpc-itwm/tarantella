import tensorflow as tf
from tensorflow import keras

import tarantella as tnt

# Load MNIST dataset
[...]

# Create Tarantella model from a `keras.Model`
model = tnt.Model(lenet5_model_generator())

# Instantiate a Tarantella optimizer from a `keras.Optimizer`
optimizer = tnt.Optimizer(keras.optimizers.SGD(learning_rate=1e-3))

# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Distribute datasets
distributed_train = tnt.data.Dataset(dataset = train_dataset)
train_dataset = distributed_train.distribute_dataset_across_ranks(is_training = True)

for epoch in range(epochs):
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

    # Display metrics at the end of each epoch
    if tnt.is_master_rank():
      print(f"Epoch {epoch}/{epochs} - Training accuracy: {train_acc_metric.result()}")

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()
