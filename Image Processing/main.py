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

import numpy as np
import tensorflow as tf
import argparse
import time
import cv2
import requests
from multiprocessing import Process, Queue
import multiprocessing



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'C:\\Users\\KDM\\PycharmProjects\\CNN\\CNNcifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'C:\\Users\\KDM\\PycharmProjects\\CNN\\cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def part_save(img_trim, i, j):
    img_trim = cv2.resize(img_trim, (32, 32), interpolation=cv2.INTER_AREA)
    cv2.imwrite('C:\\Users\\KDM\\PycharmProjects\\CNN\\image_kdm\\test_data\\Capture_image\\part_frame%d_%d.jpg' % (i,j), img_trim)

def inference(images):

# conv1
    NUM_CLASSES=4
    with tf.variable_scope('conv1') as scope:
        # shape : 5*5*3 필터를 64 장 사용한다는 뜻
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))  # 필터 64장 사용했기에 64개 biases
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1
        # 3*3 필터를 통해 pooling 수행한다.
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3   --> 새로 추가한 것이다 . shape 설정 어떻게 해야될지 모르겠다.
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    # norm3
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm3')
    # pool3
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4--> 새로 추가한 것이다.shape 설정 어떻게 해야될지 모르겠다.
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    # norm4
    norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm4')
    # pool4
    pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # local5
    with tf.variable_scope('local5') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool4, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local6
    with tf.variable_scope('local6') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local6 = tf.nn.relu(tf.matmul(local5, weights) + biases, name=scope.name)
        # dropout 추가해주었다.
        # drop_local4 = tf.nn.dropout(local4, keep_prob=0.7)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local6, weights), biases, name=scope.name)

    return softmax_linear

def classifier(result,result2, ARG, ARG2, ARG3, ARG4):   #result = 좌표, result2 = 사진
   # global count_class
    MOVING_AVERAGE_DECAY = 0.9999
    print('classifier start')

    with tf.Graph().as_default() as g:

        inputs = tf.placeholder(dtype=tf.float32, shape=(1, 32, 32, 3))

        logit = inference(inputs)
        argmax = tf.argmax(logit, axis=1)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        cnt = None

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            while True:

                count = ARG.get()
                time.sleep(1)
                lists = []

                if not result.empty() :
                    s__ = time.time()
                    objs = result.get()#좌표
                    #print(objs)
                    frame=result2.get()
                    temp=0
                    if len(objs) > 0:
                        for obj in objs:
                            temp=temp+1
                            x=obj[0]
                            y=obj[1]
                            w=obj[2]
                            h=obj[3]
                            #print('x y w h %d %d %d %d'%(x,y,w,h))
                            img_trim = frame[y:y + h, x:x + w]
                            resized_img = cv2.resize(img_trim,(32, 32))
                            cv2.imwrite(
                                'C:\\Users\\KDM\\PycharmProjects\\CNN\\Image_kdm\\test_data\\Capture_image\\test%d_%d.jpg'%(count,temp), resized_img
                            )

                            im_array = np.array(resized_img)
                            im_arrays = np.array([im_array])

                            result_ = sess.run(argmax, feed_dict={inputs: im_arrays})
                            logit_data = sess.run(logit, feed_dict={inputs:im_arrays})    # 평가 logit 데이터
                                                    # 분산값 확인해서 사람의 정도를 판단하려면 고민해봐야함.
                            lists.append(result_[0])      # 0123 리스트..  3인 부분 찾아서 make Rect
                            print(logit_data[0])
                            if result_[0] == 0:
                                print("bird")
                            elif result_[0] == 1:
                                print("cat")

                            elif result_[0] == 2:
                                print("dog")
                            else:
                                print("people")
                                print("색칠공부시작")
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    s_ = time.time() - s__
                    print(s_)
                    ARG2.put(lists)
                    ARG3.put(count)
                    ARG4.put(frame)


            #result.clear()

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)

def read_3frame(camera):
    # Read three images first:
    t_minus = cv2.cvtColor(camera.read()[1], cv2.COLOR_RGB2GRAY)
    t = cv2.cvtColor(camera.read()[1], cv2.COLOR_RGB2GRAY)
    t_plus = cv2.cvtColor(camera.read()[1], cv2.COLOR_RGB2GRAY)
    return t_minus, t, t_plus

def option_set():
    # 터미널 혹은 cmd 에서 python -m  파이썬코드.py -v 동영상이름   으로 하면 저장된 동영상에 대해서 motion detect 수행
    # ex.  python -m  test.py -a 8000  를 하면 검출하는 모션의 픽셀크기가 최소 8000 이다.
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to the video file")
    ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
    args = vars(ap.parse_args())
    return args

def save_and_request(ARG2, ARG3, ARG4): # full_frame save and request
    while True :

        img_trim = ARG4.get()
        temp = ARG2.get()
        count = ARG3.get()

        if 3 in temp :   # 3 = 사람, empty 하다면 사람으로 판단안된것.
            print("count =", count)
            #img_trim = cv2.resize(img_trim, (640, 480), interpolation=cv2.INTER_AREA)
            cv2.imwrite('C:\\Users\\KDM\\PycharmProjects\\CNN\\Image_kdm\\test_data\\Full_image\\full_frame%d.jpg' % (count),img_trim)
            url = 'https://recog.mybluemix.net/create'  # http://192.168.0.24:3000/create
            headers = {'content-type': 'image/jpg'}
            files = {'files': ('imageFile', open('Image_kdm\\test_data\\Full_image\\full_frame%d.jpg' % (count), 'rb'), headers)}
            requests.post(url, files=files)
            print('Image_kdm\\test_data\\Full_image\\full_frame%d.jpg' % (count))

def video_play(result,result2, ARG):
    print('video play')
    url = 'http://192.168.0.14:3000/create'
    # 현재 설정
    # fps=15 , video size = 640 *480
    # 3 frame 당 한 번씩 motion check

    args = option_set()

    # count frame number
    fps = 1
    # count frame number
    count = 0

    # save video --> not complete
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter("/home/pi/diff_5.avi", fourcc, 15, (640, 480))

    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)
        # 영상(웹캠 or 파이카메라) fps 설정
        camera.set(cv2.CAP_PROP_FPS, 15)

        # Read three images first:
        t_minus, t, t_plus = read_3frame(camera)

    # otherwise, we are reading from a video file
    else:
        camera = cv2.VideoCapture(args["video"])
        t_minus, t, t_plus = read_3frame(camera)

    # initialize the first frame in the video stream
    firstFrame = None

    # loop over the frames of the video
    while True:
        (grabbed, frame) = camera.read()

        if (count % 1 == 0):
            if not grabbed:
                break

            # resize the frame
            frame = cv2.resize(frame, (640, 480))
            temp = diffImg(t_minus, t, t_plus)
            temp = cv2.GaussianBlur(temp, (17, 17), 0)
            temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # count2 --> 해당 프레임 내에서의 모션이 감지된 객체의 수

            rect_info = []

            if (count % 1 == 0):
                (_, cnts, _) = cv2.findContours(temp.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 0.000x
                lists=[]
                num_part = 0

                for c in cnts:  # cnt 만큼 캡쳐 __ part_frame
                    # if the contour is too small, ignore it
                    if cv2.contourArea(c) < args["min_area"] or cv2.contourArea(c) > 5000:
                        continue

                    # compute the bounding box for the contour, draw it on the frame,
                    # and update the text
                    (x, y, w, h) = cv2.boundingRect(c)

                    # rect_info : part_frame 좌표
                    rect_info.extend([[x, y, w, h]])

                    # save each rectangle
                    img_trim = frame[y:y + h, x:x + w]

                    part_save(img_trim, count, num_part)
                    num_part += 1
                print('part 객체 수: ', len(rect_info))

                if len(rect_info) != 0:         # 좌표 = result, 사진 저장=result2 queue
                    result.put(rect_info)
                    result2.put(frame)
                    ARG.put(count)


                #    lists = classifier(result,result2, ARG)
                 #   ARG.put(lists)  # man_frame == full_frame, 사람있는 것

                #if 3 in lists :
                #    save_and_request(man_frame,count)

                # show the frame and record if the user presses a key
                cv2.imshow("original", frame)

            # Read next image
            t_minus = t
            t = t_plus

            if camera.read()[1] != '\0':  # next frame out !!!!!!!!!!!!!!!!
                count = count + 1
                t_plus = cv2.cvtColor(camera.read()[1], cv2.COLOR_RGB2GRAY)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

def main():
    result = multiprocessing.Queue()
    result2 = multiprocessing.Queue()
    ARG = multiprocessing.Queue()
    ARG2 = multiprocessing.Queue()
    ARG3 = multiprocessing.Queue()
    ARG4 = multiprocessing.Queue()

    # process1 - -
    pro1 = Process(name = "pro1", target = video_play, args = (result,result2, ARG))

    # process2 - -
    pro2 = Process(name = "pro2", target = classifier, args = (result ,result2, ARG, ARG2, ARG3, ARG4))

    # process3 - -
    pro3 = Process(name = "pro3", target = save_and_request, args = (ARG2, ARG3, ARG4))

    pro1.start()
    pro2.start()
    pro3.start()

    pro1.join()
    pro2.join()
    pro3.join()


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time() - start
    print(end)