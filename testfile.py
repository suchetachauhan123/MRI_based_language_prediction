from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
from datetime import datetime
import time
from decimal import *

import numpy as np
from numpy import swapaxes
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy
from scipy.stats import pearsonr
import tensorflow as tf

import nibabel as nib
import nilearn
from nilearn.image import resample_img

NUM_of_SLICES = 61
IMAGE_SIZE_ROW = 61
IMAGE_SIZE_COL = 73
NUM_CHANNELS = 1			# RGB color channel
n_classes = 1
SEED = None  # Set to None for random seed.
BATCH_SIZE = 14
EVAL_BATCH_SIZE = 1
EVAL_FREQUENCY = 7  # Number of steps between evaluations.

tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

def main(argv=None):  # pylint: disable=unused-argument

  # upload a test file here:
  path = os.getcwd()
  directory = 'network'
  lr = 0.001
  h = 500
  new_path= os.path.join(path, directory)

  img = nib.load('Example_lesion.nii')
  target_affine = np.array([[-3,-0,-0,90], [-0,3,-0,-126], [0,0,3,-72], [0,0,0,1]])
  new_img = nilearn.image.resample_img(img,target_affine=target_affine, target_shape=(61,73,61))
  a = np.array(new_img.dataobj)
  y = np.swapaxes(a,0,2)
  z = np.swapaxes(y,1,2)
  zz = z.reshape(61,4453)
  np.savetxt('TestFile.txt', zz.reshape(1, 271633), delimiter=',', fmt='%d')

  #data = np.loadtxt(os.path.join(new_path,'../Data/train/26.txt'), delimiter=',')
  #test_data = data.reshape(1,271633)
  test_data = zz.reshape(1,271633)
  print ('test_data.shape:',test_data.shape, len(test_data))

  '''inp = np.load(os.path.join(new_path,'../Data/train.npz'))['input']
  tar = np.load(os.path.join(new_path,'../Data/train.npz'))['labels']

  print (inp.shape)
  idx = np.random.permutation(inp.shape[0])
  inp = inp[idx]
  tar = tar[idx]

  batch_xs = inp
  batch_ys = tar

  # Extract it into np arrays.
  train_data = batch_xs 
  train_labels = batch_ys 

  num_epochs = 50
  train_size = train_labels.shape[0]
  '''  
  train_data_node = tf.placeholder(tf.float32, shape=(None, NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.float32, shape=(None, n_classes))
  
  conv1_weights = tf.Variable(
      tf.truncated_normal([3, 3, 3, NUM_CHANNELS, 4],  # 3x3 filter, depth 4.
                          stddev=0.01,
                          seed=SEED))
  conv1_biases = tf.Variable(tf.zeros([4]))

  fc1_weights = tf.Variable(tf.truncated_normal([8 * ((IMAGE_SIZE_ROW // 8)+1) * ((IMAGE_SIZE_COL // 8)+1) * 4, 500], stddev=0.01, seed=SEED))   

  print ('fc1 weights : NUM_of_SLICES * IMAGE_SIZE_ROW * IMAGE_SIZE_COL * 64')
  print (8, (IMAGE_SIZE_ROW //8)+1, (IMAGE_SIZE_COL // 8)+1, fc1_weights.get_shape())
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[500]))
  fc2_weights = tf.Variable(tf.truncated_normal([500, n_classes], stddev=0.01, seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

  def model(data):
    """The Model definition."""
    print ('conv-1 input')
    print (data)
    conv = tf.nn.conv3d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1, 1],
                        padding='SAME')

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool3d(relu,
                          ksize=[1, 9, 9, 9, 1],
                          strides=[1, 8, 8, 8, 1],
                          padding='SAME')
    print ('conv-1 output')
    print (pool)
    pool_shape = pool.get_shape().as_list()
    print (pool_shape)
    reshape = tf.reshape(pool, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3] * pool_shape[4]])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    return tf.nn.sigmoid(tf.matmul(hidden, fc2_weights) + fc2_biases), hidden

  logits, fp1 = model(train_data_node)

  loss = tf.reduce_mean(tf.pow(train_labels_node - logits, 2))
  
  '''batch = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
	lr,                # Base learning rate.
    	batch * BATCH_SIZE,  # Current index into the dataset.
    	train_size,          # Decay step.
    	0.95,                # Decay rate.
    	staircase=True)
  
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = batch) # Adam Optimizer
  '''
  ##############################################################

  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    #total_batches = data.shape[0] // EVAL_BATCH_SIZE
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, n_classes), dtype=np.float32)
    eval_fp1 = np.ndarray(shape=(size, h), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :], eval_fp1[begin:end, :] = sess.run([logits, fp1], feed_dict={train_data_node: data[begin:end, ...]})

      else:
        batch_predictions, batch_eval_fp1 = sess.run([logits, fp1], feed_dict={train_data_node: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
        eval_fp1[begin:, :] = batch_eval_fp1[begin - size:, :] 

    return predictions, eval_fp1

  def acc_loss(predictions, labels):
        loss = np.power(predictions - labels, 2)
        return np.mean(loss)

  def PearsonCorr(predictions, labels):
	r_row, p_value = pearsonr(predictions, labels)
	r_row = scipy.square(r_row)
	return r_row, p_value

  saver = tf.train.Saver()
  start_time = time.time()

  with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    print('Initialized!')

    step_end = 0
    flag = 0
    validation_loss_prev = 1
    #no_of_batches = math.ceil(train_size / float(BATCH_SIZE))
    i = 1
    count_loss = 0
    count_epoch = 0
    
    #batch_data = train_data.reshape(len(train_data), NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS)
    #print (batch_data.shape)
    #batch_labels = train_labels.reshape(len(train_data), n_classes)

    testing_data = test_data.reshape(1, NUM_of_SLICES, IMAGE_SIZE_ROW, IMAGE_SIZE_COL, NUM_CHANNELS)
 
    '''f_loss=open(os.path.join(new_path,'loss.txt'), 'w')

    # Loop through training steps.
    for step in xrange((int(num_epochs * train_size) // BATCH_SIZE)):
      #offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE + 1)
      offset = (step * BATCH_SIZE) % train_size
      print('offset:%d, offset + BATCH_SIZE:%d' %(offset, offset + BATCH_SIZE))
      inp_data = batch_data[offset:(offset + BATCH_SIZE),...]
      inp_labels = batch_labels[offset:(offset + BATCH_SIZE),...]
      feed_dict = {train_data_node: inp_data, train_labels_node:inp_labels}
      start_time = time.time()
      _, l, predictions, lr, train_fp1 = sess.run([optimizer, loss, train_prediction, learning_rate, fp1], feed_dict = feed_dict)
      duration = time.time() - start_time
      saver.save(sess, os.path.join(new_path,"convNet.ckpt"))

      if step == (i * no_of_batches) - 1:
	      idx = np.random.permutation(batch_data.shape[0])
              batch_data = batch_data[idx]
	      batch_labels = batch_labels[idx]

              print('%s: Step %d (epoch %.2f), %.3f sec/epoch' % (datetime.now(), step, float(step+1) * BATCH_SIZE / train_size, duration))
    	      predictions, train_fp1 = eval_in_batches(batch_data, sess)
              tr_loss = acc_loss(predictions, batch_labels)
              print('Training loss: %.6f, learning rate: %.10f' % (tr_loss, lr))
              print('-------------------------------------------------------------------------------------------------------')
	      f_loss.write('{:f}\n'.format(tr_loss))

              i = i + 1

      elif step % EVAL_FREQUENCY == 0:
        print('Step %d (epoch %.2f)' % (step, float(step+1) * BATCH_SIZE / train_size))
    f_loss.close()   
    '''
    saver.restore(sess, os.path.join(new_path,"convNet.ckpt"))
    '''print('****************************************** TRAINING PREDICTIONS: ******************************************')
    predictions, train_fp1 = eval_in_batches(batch_data, sess)
    tr_loss = acc_loss(predictions, batch_labels)
    np.savetxt(os.path.join(new_path,'training_prediction.txt'), predictions, fmt='%.4f')
    np.savetxt(os.path.join(new_path,'training_labels.txt'), batch_labels, fmt='%.4f')
    print('Training loss: %.4f' % (tr_loss))
    p_corr, p = PearsonCorr(predictions, batch_labels)
    print ('Sq_Pearson_corr: %.4f, p-value: %.10f' %(p_corr, p))
    '''
    # Finally print the result!
    print('*********************************************** TESTING new image PREDICTIONS: ******************************************')
    start_time = time.time()
    test_preds, test_fp1 = eval_in_batches(testing_data, sess)
    duration = time.time() - start_time
    print('%.3f sec in testing new image, test_preds: %.4f' %(duration, test_preds))

    np.savetxt(os.path.join(new_path,'testing_prediction.txt'), test_preds, fmt='%.4f')

    if FLAGS.self_test:
      assert full_loss(test_preds, testing_labels) == 0.0, 'expected 0.0 test_accuracy, got %.2f' % (full_loss(test_preds, testing_labels),)

if __name__ == '__main__':
    tf.app.run()
