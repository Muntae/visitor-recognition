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
import cv2
import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'C:\\Users\\SOL\\PycharmProjects\\untitled1\\cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\Users\\SOL\\PycharmProjects\\untitled1\\cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60*1, # 60 초 동안 멈춘 후 다시 eval_once 동작
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 4000, #test 할 data 개수 입력
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, logits, labels, top_k_op,# summary_op,
                images,eval_count):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint , 내가 학습시킨 모델을 복구한다.
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      thread_count=0
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
        thread_count+=1
      print ('thread count : ',thread_count)

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.     # 맞게 평가한 횟수
      total_sample_count = num_iter * FLAGS.batch_size                # 총 평가 수
      step = 0
      while step < num_iter and not coord.should_stop():
        # step 이 진행되면서  배치도 이동한다.
        """
        sess.run 할 때도 logit, label 을 먼저 한 뒤에 top_k_op 해야 정상적인 결과가 나옴.
        
        """
        #test_image --> batch_size 길이의 배열
        logit,Label,test_image,predictions = sess.run([logits,labels,images,top_k_op])
        # 맞게 예측한 것의 수 count
        true_count += np.sum(predictions)
        tf.summary.image('eval:%d _step%d' %(eval_count,step), test_image, max_outputs=30)
        # Logit = 각 label에 대한 확률.
        # Label = label
        print('eval         :', eval_count)
        print('step         :', step)

        #텐서보드를 통해 비교해볼 때 아래 주석 해제
        """
        step2 = 0
        cifar10classes = ['bird','cat','dog','people']
        while (True):
            classification = sess.run(tf.argmax(logit[step2], 0))
            print('logit[%d]     :' % step2, logit[step2])
            print('logit[%d]s predict label is %d  :' % (step2, classification), cifar10classes[classification]),
            print('logit[%d]s    true label is %d : '%(step2, Label[step2]), cifar10classes[Label[step2]])
            print('--------------------------------------------------------')
            step2 = step2 + 1
            if (step2 == FLAGS.batch_size):
                break
        
        """
        step +=1
        #중단점 설정
        #if(step==10):
        #    break
      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary_op=tf.summary.merge_all()#Merges all summaries collected in the default graph.
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop() # 스레드에 정지 요청
    coord.join(threads, stop_grace_period_secs=10) #다음 코드를 진행하기전에, Queue Runner의 모든 쓰레드들이 정지될때 까지 기다리는 코드이다.
    #stop_grace_period_secs = 10 은 coord.request_stop 수행 후 10초 뒤에 join 함수 수행
def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data) # input 받아오기 -> batch 크기로
    #tf.summary.image('images', images, max_outputs=30)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # logtis, top_k_op 등은 아직 세션이 시작되지 않았기 때문에 연산이 되기 전이다.
    logits = cifar10.inference(images)  # 평가 받을 데이터로 만든것. batch 크기 기준,

    # Calculate predictions.
    # 우리는 예측이 실제 라벨과 정확히 일치할 경우만 올바르다고 기록하기 위해 K의 값(3번째 인자)을 1로 두었습니다.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)  # 평가 받을 데이터로 만든 것. batch크기 기준


    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
    eval_count= 0
    while True:
      eval_once(saver, summary_writer, logits, labels, top_k_op, #summary_op,
                  images,eval_count)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
      if eval_count==0:
          break
      eval_count +=1


def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()