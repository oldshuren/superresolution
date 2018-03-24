import numpy as np
import tensorflow as tf

import math
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--input_data', type=str,
                      help='Train data file')
parser.add_argument('--model_dir', type=str,
                      help='The directory where the saved mode will be stored')

FLAGS, _ = parser.parse_known_args()

def preprocess(img) :
  # Normalize it,  rescale to [-1,1]
  ret = img - np.min(img)
  ret = ret / np.max(ret)
  ret = ret - 0.5
  ret = ret * 2.0
  return ret

with open(FLAGS.input_data, 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter=' ', skipinitialspace = True)
  numbers = [list(map(float,rec)) for rec in reader]
  expected = [x[:2] for x in numbers]
  image = [preprocess(x[2:]) for x in numbers]

  saved_model_predictor = tf.contrib.predictor.from_saved_model(export_dir=FLAGS.model_dir)

  result = saved_model_predictor({'image':image})
  
  x = result['x']
  y = result['y']

  # accumulate error
  errors = np.array([])
  for pred_x, pred_y, ev in zip(x, y, expected):
    expected_x, expected_y = ev
    err = math.sqrt((pred_x - expected_x)*(pred_x-expected_x) + (pred_y - expected_y)*(pred_y - expected_y))
    print('predicted=({},{}) expected=({:.0f},{:.0f}), error={:.2f}'.format(pred_x,pred_y, expected_x, expected_y, err))
    errors = np.append(errors, err)

  print ('error mean={}, variance={}, max={}'.format(np.mean(errors), np.var(errors), np.amax(errors)))

