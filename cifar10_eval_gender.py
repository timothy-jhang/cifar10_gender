# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.
Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.
Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.
Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.
http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train_0.01_256_hsv',
tf.app.flags.DEFINE_string('checkpoint_dir', './cifar10_train',
                           """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_integer('eval_interval_secs',  60*5,
tf.app.flags.DEFINE_integer('eval_interval_secs',  5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(sess, saver, summary_writer, top_k_op, summary_op, global_step, coord):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """

  num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
  true_count = 0  # Counts the number of correct predictions.
  total_sample_count = num_iter * FLAGS.batch_size
  step = 0
  print('num_iter=', num_iter)
  while (step < num_iter) :
    predictions, summary = sess.run([top_k_op,summary_op])
    true_count += np.sum(predictions)
    step += 1
    summary_writer.add_summary(summary, step)
      # Compute precision @ 1.
  precision = true_count / num_iter
  print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
  return precision 

def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # Build inference Graph.
    print('>>>>> input original size = ',images)
#   in case of images less than or larger than 227x227
#   images = tf.image.resize_images(images, [227,227] )
#   print('>>>>> input resized = ',images)
    logits = cifar10.inference(images,1.0)

    # Calculate accuracy.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    top_k_op = tf.cast(top_k_op, tf.int32)
    acc_batch = tf.reduce_sum(top_k_op)
    tf.summary.scalar(name="acc_batch", tensor=acc_batch/FLAGS.batch_size)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    gpu_options = tf.GPUOptions(allocator_type='BFC',allow_growth=True)
    with tf.Session(config=tf.ConfigProto( allow_soft_placement=True,gpu_options=gpu_options) ) as sess:
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return

      # Creates a variable to hold the global_step.
      global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)


      coord = tf.train.Coordinator()
      # Start the queue runners.
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))
      try:
        no_rep = 20.0
        sumprec = 0.0 
        for i in range(no_rep): # 100 times repetition
          prec = eval_once(sess, saver, summary_writer, top_k_op, summary_op, global_step, coord )
          print('precision = ', prec)
          sumprec += prec
        print('average precision = ', sumprec / no_rep)
      except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
