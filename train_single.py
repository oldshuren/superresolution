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
parser.add_argument('--num_parallel_calls', type=int,
                      default=5,
                      help='num of parallel calls to process input')
parser.add_argument('--train_epochs', type=int,
                      default=100,
                      help='Number of Epoch')
parser.add_argument('--epochs_per_eval', type=int,
                      default=1,
                      help='The number of training epochs to run between evaluations')
parser.add_argument('--num_hidden', type=int,
                      default=1,
                      help='Number of Hidden layers')
parser.add_argument('--hidden_size', type=int,
                      default=128,
                      help='Size of hidden layers')
parser.add_argument('--data_dir', type=str,
                      default='./data/generated',
                      help='Train data directory')
parser.add_argument('--model_dir', type=str,
                      default='/tmp/superresolution',
                      help='The directory where the mode will be stored')

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
      

  return image, {'x':tf.one_hot(x, _NUM_SUBPIXEL), 'y':tf.one_hot(y, _NUM_SUBPIXEL)}

"""Input function which provides batches for train or eval."""
def input_fn(is_training, data_dir, batch_size, num_epochs=1,num_parallel_calls=1):
  dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir))

  #if is_training:
  dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value, is_training),
                          num_parallel_calls=num_parallel_calls)
  dataset = dataset.prefetch(batch_size)

  #if is_training:
  # When choosing shuffle buffer sizes, larger sizes result in better
  # randomness, while smaller sizes have better performance.
  dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  #iterator = dataset.make_one_shot_iterator()
  #return iterator.get_next()
  return dataset

# Create model
# Hpyer Parameters
#learning_rate = 0.001
learning_rate = 0.01
training_epochs = FLAGS.train_epochs

# Network Parameters
n_input = 7*7
num_hidden = FLAGS.num_hidden
hidden_size = FLAGS.hidden_size
n_output = _NUM_SUBPIXEL

# create weight and bias first

def network(x, is_training=True):
  #print('network input={}'.format(x))
  layer_input = x
  layer_output = x
  for i in range(num_hidden):
    layer_output_size = hidden_size
    # Hidden layer with RELU activation
    layer_output = tf.layers.dense(inputs=layer_input, units=layer_output_size)
    layer_output = tf.layers.batch_normalization(inputs=layer_output, training=is_training)
    layer_output = tf.nn.relu(layer_output)
    layer_input = layer_output
        
  # Output layer with linear activation
  # we have two output x,y

  x = tf.layers.dense(inputs=layer_output, units=n_output)
  y = tf.layers.dense(inputs=layer_output, units=n_output)

  return x, y

def model_fn(features, labels, mode, params):
  #is_training = mode == tf.estimator.ModeKeys.TRAIN
  is_training = True
  #print('model_fn features={}'.format(features))
  if mode == tf.estimator.ModeKeys.PREDICT:
    images = features['image']

    pred_x, pred_y  = network(images, is_training=is_training)
    pred_x_val = tf.argmax(pred_x, axis=1)
    pred_y_val = tf.argmax(pred_y, axis=1)
    predictions = {
      'x': pred_x_val,
      'y': pred_y_val
    }
    export_outputs = {
      "serving_default": tf.estimator.export.PredictOutput(predictions)}
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
  
  pred_x, pred_y  = network(features, is_training=is_training)
  pred_x_val = tf.argmax(pred_x, axis=1)
  pred_y_val = tf.argmax(pred_y, axis=1)
  # Generate a summary node for the images
  tf.summary.image('images', tf.reshape(features,[-1,7,7,1]), max_outputs=6)

  expected_x = labels['x']
  expected_y = labels['y']

  expected_x_val = tf.argmax(expected_x, axis=1)
  expected_y_val = tf.argmax(expected_y, axis=1)

  #  err is the distance of predicted coordinates to the expected coordinates
  x_err = tf.cast(pred_x_val - expected_x_val, dtype=tf.float32)
  y_err = tf.cast(pred_y_val - expected_y_val, dtype=tf.float32)
  err = tf.sqrt(tf.square(x_err) + tf.square(y_err))
  err_max = tf.reduce_max(x_err)
  err_mean, err_variance = tf.nn.moments(err,0)

  metrics = {
    'max err':tf.metrics.mean(err_max),
    'err mean':tf.metrics.mean(err_mean),
    'err variance':tf.metrics.mean(err_variance),
  }

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  loss_x = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_x, labels=expected_x))
  loss_y = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_y, labels=expected_y))
  cross_entropy = loss_x + loss_y


  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)
  tf.identity(loss_x, 'loss_x')
  tf.identity(loss_y, 'loss_y')
  tf.identity(cross_entropy, 'loss')

  # Add weight decay to the loss.
  #loss = cross_entropy + weight_decay * tf.add_n(
  #    [tf.nn.l2_loss(v) for v in tf.trainable_variables()
  #     if loss_filter_fn(v.name)])
  loss = cross_entropy
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    #learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    #tf.identity(learning_rate, name='learning_rate')
    #tf.summary.scalar('learning_rate', learning_rate)

    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Batch norm requires update ops to be added as a dependency to train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step=global_step)
  else:
    train_op = None

  #accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
  #metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  #tf.identity(accuracy[1], name='train_accuracy')
  #tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics)

# Set up a RunConfig to only save checkpoints once per training cycle.
run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
classifier = tf.estimator.Estimator(
  model_fn=model_fn, model_dir=FLAGS.model_dir, config=run_config,
  params={
    'batch_size': FLAGS.batch_size,
  })

tf.logging.set_verbosity(tf.logging.INFO)

for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
  tensors_to_log = {
    'x loss': 'loss_x',
    'y loss': 'loss_y',
  }

  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100)

  print('Starting a training cycle.')

  def input_fn_train():
    return input_fn(True, FLAGS.data_dir, FLAGS.batch_size,
                    FLAGS.epochs_per_eval, FLAGS.num_parallel_calls)

  classifier.train(input_fn=input_fn_train, hooks=[logging_hook])

  print('Starting to evaluate.')
  # Evaluate the model and print results
  def input_fn_eval():
    return input_fn(False, FLAGS.data_dir, FLAGS.batch_size,
                    1, FLAGS.num_parallel_calls)

  eval_results = classifier.evaluate(input_fn=input_fn_eval)
  print(eval_results)

print('export saved model')
inputs = {"image": tf.placeholder(shape=[None, 7*7], dtype=tf.float32)}
serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)

export_dir = classifier.export_savedmodel(
  FLAGS.model_dir,
  serving_input_receiver_fn=serving_input_receiver_fn)
print ('model exported to {}'.format(export_dir))
