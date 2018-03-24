import numpy as np
from oct2py import octave
import tensorflow as tf

import argparse
import sys
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str,
                      help='Output test file')
parser.add_argument('--sample_batch', type=int,
                    default=10,
                    help='Number of batches of the test sample generated')
FLAGS, _ = parser.parse_known_args()

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

num_of_subpixel = 10
points = np.zeros((num_of_subpixel*num_of_subpixel,2), dtype=np.int32)

for i in range(num_of_subpixel):
  for j in range(num_of_subpixel):
    points[i*num_of_subpixel+j][0] = i
    points[i*num_of_subpixel+j][1] = j

points_length = num_of_subpixel * num_of_subpixel

def main():
  if FLAGS.output is not None:
    print('Generate test samples to {}'.format(FLAGS.output))
    if sys.version_info[0] == 3:
      f = open(FLAGS.output, 'w', newline='')
    else:
      f = open(FLAGS.output, 'w')
    writer = csv.writer(f, delimiter=' ')
    ret = octave.GenerateSingle(points, FLAGS.sample_batch)
    for i in range(ret.shape[0]):
      img=ret[i].reshape(49)
      x =  points[i % points_length][0]
      y =  points[i % points_length][1];
      row = [x, y] + np.ndarray.tolist(img)
      writer.writerow(row)
    return

  num_per_postion = 1000
  image_seqno = 0

  npp = 0
  writer = tf.python_io.TFRecordWriter('generated/train-%04d.rec' % npp)
  ret = octave.GenerateSingle(points, num_per_postion)
  data_written = False

  for i in range(ret.shape[0]):
    img=ret[i].reshape(49)
    #img=img/img.max()
    x =  points[i % points_length][0]
    y =  points[i % points_length][1];
    #print('x={},y={}, img={}'.format(x,y, img))
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/seqno': _int64_feature(image_seqno),
      'image/height': _int64_feature(7),
      'image/width': _int64_feature(7),
      'image/subpixel/x': _int64_feature(x),
      'image/subpixel/y': _int64_feature(y),
      'image/encoded': _float_feature(img.tolist())}))
    image_seqno += 1
    writer.write(example.SerializeToString())
    data_written = True
    if image_seqno % points_length == 0:
      print('output to generated/train-%04d.rec' % npp)
      writer.close()
      npp += 1
      writer = tf.python_io.TFRecordWriter('generated/train-%04d.rec' % npp)
      data_written = False

  if data_written:
    print('output to generated/train-%04d.rec' % npp)
    writer.close()

  num_per_validation = 100

  image_seqno = 0
  npp = 0
  writer = tf.python_io.TFRecordWriter('generated/validate-%04d.rec' % npp)
  ret = octave.GenerateSingle(points, num_per_validation)
  data_written = False

  for i in range(ret.shape[0]):
    img=ret[i].reshape(49)
    #img=img/img.max()
    x =  points[i % points_length][0]
    y =  points[i % points_length][1];
    #print('x={},y={}, img={}'.format(x,y, img))
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/seqno': _int64_feature(image_seqno),
      'image/height': _int64_feature(7),
      'image/width': _int64_feature(7),
      'image/subpixel/x': _int64_feature(x),
      'image/subpixel/y': _int64_feature(y),
      'image/encoded': _float_feature(img.tolist())}))
    image_seqno += 1
    writer.write(example.SerializeToString())
    data_written = True
    if image_seqno % points_length == 0:
      print('output to generated/validate-%04d.rec' % npp)
      writer.close()
      npp +=1
      writer = tf.python_io.TFRecordWriter('generated/validate-%04d.rec' % npp)
      data_written = False

  if data_written:
    print('output to generated/validate-%04d.rec' % npp)
    writer.close()

if __name__ == "__main__":
    main()
