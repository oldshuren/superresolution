import numpy as np
import tensorflow as tf

import os
import sys
import imageio

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500
_NUM_TRAIN_FILES = 1000
_NUM_SUBPIXEL = 10
def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%04d.rec' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return ['validate.rec']

def record_parser(value, is_training):
  """Parse an Simulated Image record from `value`."""
  print('record_parser')
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

  image = tf.reshape(image, shape=[49])
  # Normalize it,  rescale to [-1,1] instead of [0, 1)
  #image = tf.div(image, tf.reduce_max(image))
  image = image - tf.reduce_min(image)
  image = image/tf.reduce_max(image)
#  image = tf.subtract(image, 0.5)
#  image = tf.multiply(image, 2.0)
  seqno = tf.cast(
      tf.reshape(parsed['image/seqno'], shape=[]),
      dtype=tf.int32)
  x = tf.cast(
      tf.reshape(parsed['image/subpixel/x'], shape=[]),
      dtype=tf.int32)
  y = tf.cast(
      tf.reshape(parsed['image/subpixel/y'], shape=[]),
      dtype=tf.int32)
      
  x = tf.one_hot(x, _NUM_SUBPIXEL)
  y = tf.one_hot(y, _NUM_SUBPIXEL)
  xy = tf.stack([x,y])
  xy = tf.reduce_max(xy, 0)
  cb = tf.concat([x,y], 0)
  return image, x, y, xy, cb

batch_size = 4
is_training = True
data_dir = './generated'
num_epochs = 4

"""Input function which provides batches for train or eval."""
dataset = tf.data.Dataset.from_tensor_slices(filenames(is_training, data_dir)).repeat()

if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

dataset = dataset.flat_map(tf.data.TFRecordDataset)
dataset = dataset.map(lambda value: record_parser(value, is_training),
                      num_parallel_calls=5)
dataset = dataset.prefetch(batch_size)

if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

# We call repeat after shuffling, rather than before, to prevent separate
# epochs from blending together.
dataset = dataset.repeat(num_epochs)
dataset = dataset.batch(batch_size)

iterator = dataset.make_one_shot_iterator()
fetches = iterator.get_next()

#images, label  = input_fn(True, '', 4)
num_image_output=0
image_outputed = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1):
        try:
            image, x, y, xy, cb = sess.run(fetches)
            print('image={}'.format(image))
            print('x={}'.format(x))
            print('y={}'.format(y))
            print('xy={}'.format(xy))
            print('cb={}'.format(cb))
            if image_outputed < num_image_output:
              img = val[0][0].reshape(7,7)
              x = val[2][0]
              y = val[3][0]
              # blow each pixel 10 times
              zoomed_image = np.kron(img, np.ones((10,10)))
              imageio.imwrite('image-%d-%d-%d.png'%(x,y,image_outputed), zoomed_image)
              image_outputed += 1
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # "End of dataset"
            dataset.repeat()
