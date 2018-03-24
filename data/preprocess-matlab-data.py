import numpy as np
import tensorflow as tf

import math
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str,
                    help='Train data file')
parser.add_argument('--test_data', type=str,
                    help='Test data file')
parser.add_argument('--output_dir', type=str,
                    default='generated',
                      help='The output directory for tf record')

FLAGS, _ = parser.parse_known_args()

def preprocess(img) :
  # Normalize it,  rescale to [-1,1]
  ret = img - np.min(img)
  ret = ret / np.max(ret)
  ret = ret - 0.5
  ret = ret * 2.0
  return ret


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

points_length = num_of_subpixel * num_of_subpixel

with open(FLAGS.train_data, 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=' ', skipinitialspace = True)
  numbers = [list(map(float,rec)) for rec in reader]
  expected = [x[:2] for x in numbers]
  image = [x[2:] for x in numbers]

  num_per_postion = 1000
  image_seqno = 0

  npp = 0
  writer = tf.python_io.TFRecordWriter('generated/train-%04d.rec' % npp)
  data_written = False

  for i in range(np.shape(image)[0]):
    img=image[i]
    x =  int(expected[i][0])
    y =  int(expected[i][1])
    #print('x={},y={}, img={}'.format(x,y, img))
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/seqno': _int64_feature(image_seqno),
      'image/height': _int64_feature(7),
      'image/width': _int64_feature(7),
      'image/subpixel/x': _int64_feature(x),
      'image/subpixel/y': _int64_feature(y),
      'image/encoded': _float_feature(img)}))
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

with open(FLAGS.test_data, 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=' ', skipinitialspace = True)
  numbers = [list(map(float,rec)) for rec in reader]
  expected = [x[:2] for x in numbers]
  image = [x[2:] for x in numbers]
  
  num_per_validation = 100

  image_seqno = 0
  npp = 0
  writer = tf.python_io.TFRecordWriter('generated/validate-%04d.rec' % npp)
  data_written = False

  for i in range(np.shape(image)[0]):
    img=image[i]
    x =  int(expected[i][0])
    y =  int(expected[i][1])
    #print('x={},y={}, img={}'.format(x,y, img))
    example = tf.train.Example(features=tf.train.Features(feature={
      'image/seqno': _int64_feature(image_seqno),
      'image/height': _int64_feature(7),
      'image/width': _int64_feature(7),
      'image/subpixel/x': _int64_feature(x),
      'image/subpixel/y': _int64_feature(y),
      'image/encoded': _float_feature(img)}))
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

  
