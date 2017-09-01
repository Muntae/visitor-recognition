# -*- coding: utf-8 -*-
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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

cifar10classes = ["bird", "cat", "dog", "people"]

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32  # 32*32 로 test 할 것이다.

# Global constants describing the CIFAR-10 data set.
# 변경된 dataset 개수로 수정
NUM_CLASSES = 4 #bird, cat, dog, people
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 20000  # 각 label 당  5천장씩 .
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 4000# 각 label 당  1천장씩 .

#label 파일을 처리하는 함수
def read_cifar10_label(filename_queue):

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()


    label_bytes = 1  # 2 for CIFAR-100
    record_bytes = label_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    return result


# 이미지 파일들을 처리하는 함수
def read_cifar10(filename_queue):

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    label_bytes = 0  # 2 for CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    """
    wholeFileReader 를 통해 jpg 파일들이 저장된 queue 를 가져온 후 read 하여 value 저장한다.
    result.key 에 의해 자동으로 다음 데이터를 가리켜서 알아서 넘어간다.
    decode_jpeg 함수를 통해 큐를 이미지 형태의 텐서로 만든다.
    """
    reader = tf.WholeFileReader()
    # Returns the next record (key, value pair) produced by a reader.
    result.key, value = reader.read(filename_queue)
    # reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # result.key, value = reader.read(filename_queue)

    """
    """
    result.uint8image=tf.image.decode_jpeg(value)
    return result


# train, eval 때 호출함
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # default 이미지 저장 개수 3이다, 이 함수 내에서는 하나의 batch 만큼을 리턴하는 것이므로 max_outputs 은  batch_size 로 설정하면 된다.
    tf.summary.image('images', images, max_outputs=20)

    return images, tf.reshape(label_batch, [batch_size])


# train 때 호출
def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    """

    """
    filename 큐를 생성한다.
    bird, cat, dog, people 폴더를 참조한다. 다른 cifar10 data 쓰고 싶으면 알아서 수정하면 된다.
    여기에서 사용된 train data 수 에 따라서 NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN 조정해야한다.
    filenaems = bird 이미지 5000개, ... ,  people 이미지 5000개 의 순서로 저장됨.
    """
    #read image file queue
    print('train_data dir :',data_dir)
    filenames = [
        os.path.join(data_dir, 'bird\\bird_%d.jpg' % i)
        for i in xrange(0, 5000)]
    filenames2 = [
        os.path.join(data_dir,'cat\\cat_%d.jpg' % i)
        for i in xrange(0, 5000)]
    filenames3 = [
        os.path.join(data_dir, 'dog\\dog_%d.jpg' % i)
        for i in xrange(0, 5000)]
    filenames4 = [
        os.path.join(data_dir, 'people\\people_%d.jpg' % i)
        for i in xrange(0, 5000)]

    filenames.extend(filenames2)
    filenames.extend(filenames3)
    filenames.extend(filenames4)

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    # shuffle 디폴트 값이 True 라서 False 로 변경하였다. True 이면 image, label 순서가 안맞게 됨
    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    #read label file queue
    filenames_label=[os.path.join(data_dir, 'train_label.bin')]
    filename_queue2 = tf.train.string_input_producer(filenames_label,shuffle=False)
    # Read examples from files in the filename queue.
    read_label = read_cifar10_label(filename_queue2)

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image=tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)
    # image filter section

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_label.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_label.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


# eval 할 때 호출 됨
def inputs(eval_data, data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    """
    # distorted_inputs 이랑 동작방식 거의 똑같아서 주석 생략한다.
    print('eval_data dir : ', data_dir)

    # 여기 if 문은 수정 안했음.
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        print('여기 나오면 에러')
    # eval 할 때  현재 각 종류별로 1000장씩 보유중이라서 xrange 범위 0 ,1000 이다.
    else:

        filenames = [
            os.path.join(data_dir, 'bird\\bird_%d.jpg' % i)
            for i in xrange(0, 1000)]

        filenames2 = [
            os.path.join(data_dir, 'cat\\cat_%d.jpg' % i)
            for i in xrange(0, 1000)]
        filenames3 = [
            os.path.join(data_dir, 'dog\\dog_%d.jpg' % i)
            for i in xrange(0, 1000)]
        filenames4 = [
            os.path.join(data_dir,'people\\people_%d.jpg' % i)
            for i in xrange(0, 1000)]

        filenames.extend(filenames2)
        filenames.extend(filenames3)
        filenames.extend(filenames4)
        #print('파일목록', filenames)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    #이미지 파일들이 저장된 큐
    #tf.train.string_input_producer 기본적으로 자동으로 셔플 된다고 한다.
    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)



    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    """
    read_cifar10_label 수행
    """
    # read label file queue
    filenames_label = [os.path.join(data_dir, 'test_label.bin')]
    filename_queue2 = tf.train.string_input_producer(filenames_label,shuffle=False)
    # Read examples from files in the filename queue.
    read_label = read_cifar10_label(filename_queue2)

    """
       
    """

    # Image processing for evaluation.

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_label.label.set_shape([1])


    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_label.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)



