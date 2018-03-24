import numpy as np
import tensorflow as tf

import os
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int,
                      default=256,
                      help='batch size')
parser.add_argument('--num_epoch', type=int,
                      default=100,
                      help='Number of Epoch')
parser.add_argument('--num_hidden', type=int,
                      default=1,
                      help='Number of Hidden layers')
parser.add_argument('--hidden_size', type=int,
                      default=128,
                      help='Size of hidden layers')
parser.add_argument('--data_dir', type=str,
                      default='./data/generated',
                      help='Train data directory')

FLAGS, _ = parser.parse_known_args()

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500
_NUM_TRAIN_FILES = 1000
_NUM_VALIDATE_FILES = 100

def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
      return [
        os.path.join(data_dir, 'train-%04d.rec' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
      return [
        os.path.join(data_dir, 'validate-%04d.rec' % i)
        for i in range(_NUM_VALIDATE_FILES)]

_NUM_SUBPIXEL = 10

def record_parser(value, is_training):
  """Parse an Simulated Image record from `value`."""
  keys_to_features = {
      'image/seqno': tf.FixedLenFeature([1], dtype=tf.int64,
                                         default_value=-1),
      'image/height': tf.FixedLenFeature([1], dtype=tf.int64,
                                         default_value=-1),
      'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                        default_value=-1),
      'image/subpixel/x': tf.FixedLenFeature([1], dtype=tf.int64,
                                             default_value=-1),
      'image/subpixel/y': tf.FixedLenFeature([1], dtype=tf.int64,
                                             default_value=-1),
      'image/encoded': tf.FixedLenFeature([49], dtype=tf.float32),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = parsed['image/encoded']
  # Normalize it,  rescale to [-1,1] instead of [0, 1)
  image = tf.subtract(image, tf.reduce_min(image))
  image = tf.div(image, tf.reduce_max(image))
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)

  x = tf.cast(
      tf.reshape(parsed['image/subpixel/x'], shape=[]),
      dtype=tf.int32)
  y = tf.cast(
      tf.reshape(parsed['image/subpixel/y'], shape=[]),
      dtype=tf.int32)
      

  return image, tf.one_hot(x, _NUM_SUBPIXEL), tf.one_hot(y, _NUM_SUBPIXEL)
  #return image, x, y

"""Input function which provides batches for train or eval."""
def input_fn(is_training, batch_size, data_dir, num_epochs=4):
    print ('input_fn batch_size %d' % batch_size)
    dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir)).repeat()

    #if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value, is_training),
                          num_parallel_calls=5)
    dataset = dataset.prefetch(batch_size)

    #if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    
    return iterator.get_next()

# Create model
# Hpyer Parameters
#learning_rate = 0.001
learning_rate = 0.01
training_epochs = FLAGS.num_epoch
batch_size = FLAGS.batch_size
display_step = 1

# Network Parameters
n_input = 7*7
num_hidden = FLAGS.num_hidden
hidden_size = FLAGS.hidden_size
n_output = _NUM_SUBPIXEL

# create weight and bias first

layer_input_size = n_input
print('first layer input size %d'%layer_input_size)
layer_output_size = layer_input_size

hidden_tense_layers = [None] * num_hidden
hidden_bn = [None] * num_hidden

hidden_layer_weights = [None] * num_hidden
hidden_layer_biases = [None] * num_hidden

for i in range(num_hidden):
    layer_output_size = hidden_size
    hidden_tense_layers[i] =  tf.layers.Dense(layer_output_size)
    hidden_bn[i] = tf.layers.BatchNormalization()
    hidden_layer_weights[i] =  tf.Variable(tf.random_normal([layer_input_size, layer_output_size]))
    hidden_layer_biases[i]=  tf.Variable(tf.random_normal([layer_output_size]))
    layer_input_size = layer_output_size

out_layer_denes_x = tf.layers.Dense(n_output)
out_layer_denes_y = tf.layers.Dense(n_output)

out_layer_weight_x =  tf.Variable(tf.random_normal([layer_output_size, n_output]))
out_layer_bias_x =  tf.Variable(tf.random_normal([n_output]))
out_layer_weight_y =  tf.Variable(tf.random_normal([layer_output_size, n_output]))
out_layer_bias_y =  tf.Variable(tf.random_normal([n_output]))


# train and eval network are sharing variables
def network(x, is_training=True):
    layer_input = x
    layer_output = x
    for i in range(num_hidden):
        layer_output_size = hidden_size
        # Hidden layer with RELU activation
        layer_output = hidden_tense_layers[i](layer_input)
        layer_output = hidden_bn[i].apply(layer_output, training=is_training)
        layer_output = tf.nn.relu(layer_output)
        layer_input = layer_output
        
    # Output layer with linear activation
    # we have two output y1, y2

    x = out_layer_denes_x(layer_output)
    y = out_layer_denes_y(layer_output)

    return x, y

def network1(x, is_training=True):
    layer_input = x
    layer_output = x
    for i in range(num_hidden):
        weight =  hidden_layer_weights[i]
        bias =  hidden_layer_biases[i]
        # Hidden layer with RELU activation
        layer_output = tf.add(tf.matmul(layer_input, weight), bias)
        layer_output = hidden_bn[i].apply(layer_output, training=is_training)
        #layer_output = tf.layers.batch_normalization(layer_output, training=is_training)
        layer_output = tf.nn.relu(layer_output)
        layer_input = layer_output
        
    # Output layer with linear activation
    # we have two output x, y

    x = tf.matmul(layer_output, out_layer_weight_x) + out_layer_bias_x
    y = tf.matmul(layer_output, out_layer_weight_y) + out_layer_bias_y

    return x,y

train_images,train_expected_x,train_expected_y = input_fn(True, FLAGS.batch_size, FLAGS.data_dir)

is_training = tf.placeholder(tf.bool)
# Construct model
train_pred_x, train_pred_y  = network(train_images, is_training=is_training)
train_pred_x_sm = tf.nn.softmax(train_pred_x)
train_pred_y_sm = tf.nn.softmax(train_pred_y)

# Define loss and optimizer
cost_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_pred_x, labels=train_expected_x))
cost_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=train_pred_y, labels=train_expected_y))
# mininize both
cost = cost_x+cost_y

# using eculidean distance as the cost function
#pred_x_pos = tf.nn.softmax(train_pred_x)
#pred_y_pos = tf.nn.softmax(train_pred_y)
#cost = tf.reduce_mean(tf.square( pred_x_pos - train_expected_x) + tf.square( pred_y_pos - train_expected_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#opt_x = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_x)
#opt_y = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_y)
#optimizer = tf.group(opt_x, opt_y)
#optimizer = opt_x

# Test model
test_images, test_expected_x, test_expected_y = input_fn(False, FLAGS.batch_size, FLAGS.data_dir)

test_pred_x, test_pred_y  = network(test_images, is_training)

test_pred_x_val = tf.cast(tf.argmax(test_pred_x, axis=1), dtype=tf.float32)
test_pred_y_val = tf.cast(tf.argmax(test_pred_y, axis=1), dtype=tf.float32)

test_expected_x_val = tf.cast(tf.argmax(test_expected_x, axis=1), dtype=tf.float32)
test_expected_y_val = tf.cast(tf.argmax(test_expected_y, axis=1), dtype=tf.float32)

x_err = tf.cast(test_pred_x_val - test_expected_x_val, dtype=tf.float32)
y_err = tf.cast(test_pred_y_val - test_expected_y_val, dtype=tf.float32)
err = tf.sqrt(tf.square(x_err) + tf.square(y_err))
err_max = tf.reduce_max(x_err)
err_mean, err_variance = tf.nn.moments(err,0)


# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph
num_training_samples = _NUM_TRAIN_FILES * 100
num_validation_samples = _NUM_VALIDATE_FILES * 100

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    print("start traing")
    start_time = time.time()
    for epoch in range(training_epochs):
        avg_costx = 0.
        avg_costy = 0.
        total_batch = int(num_training_samples/batch_size)
        #total_batch =  5000
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cx,cy = sess.run([optimizer, cost_x, cost_y], feed_dict={is_training:True})
            # Compute average loss
            avg_costx += cx / total_batch
            avg_costy += cy / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: {:04d} cost= {:.9f},{:.9f}".format(epoch+1, avg_costx, avg_costy))

    elapsed_time = time.time() - start_time
    print("training time {}".format(elapsed_time))
    print("********** Validating.... **************")
    total_batch = int(num_validation_samples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
      mean, variance, emax = sess.run([err_mean, err_variance, err_max], feed_dict={is_training:True}) 
      print("test error mean={:.3f}, variance={:.3f}, max={:.1f}".format(mean, variance, emax))


#    start_time = time.time()
#    ac =  sess.run(accuracy)
#    elapsed_time = time.time() - start_time
#    print("accuracy {}".format(ac))
#    print("testing time {}".format(elapsed_time))


