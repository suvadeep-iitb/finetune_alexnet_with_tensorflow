# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import pickle
from collections import Counter

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, pickle_file, batch_size, num_classes, shuffle=True):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.pickle_file = pickle_file
        self.num_classes = num_classes

        # load the data from the pickle file
        self._load_from_pickle_file()

        # number of samples in the dataset
        self.data_size = self.labels.get_shape().as_list()[0]
 
        # convert labels to one_hot representation
        self.labels = tf.one_hot(self.labels, num_classes)
     
        # create dataset
        data = Dataset.from_tensor_slices((self.img_contents, self.labels))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=self.data_size, \
                                reshuffle_each_iteration=True)

        # create a new dataset with batches of images
        data = data.batch(batch_size, drop_remainder=True)

        self.data = data


    def _load_from_pickle_file(self):
        self.img_contents, self.labels = pickle.load(open(self.pickle_file, 'rb'))
        if type(self.img_contents) is list:
            for i in range(len(self.img_contents)):
                self.img_contents[i] = convert_to_tensor(self.img_contents[i], dtype=dtypes.float32)
            self.img_contents = tf.concat(self.img_contents, axis=0)
        else:
            self.img_contents = convert_to_tensor(self.img_contents, dtype=dtypes.float32)
        self.label_list = self.labels
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)


    def _load_images(self):
        """Load images into memory."""
        img_list = []
        for i in range(self.data_size):
            feature = convert_to_tensor(np.load(self.img_paths[i]))
            img_list.append(feature)
        self.img_contents = img_list


    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split()
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

    def get_acc_split_weights(self, caffe_class_ids):
        class_counter = Counter(self.label_list)
        class_ids = np.array(list(class_counter.keys()))
        class_freqs = np.array(list(class_counter.values()))
        sorted_idx = np.argsort(-class_freqs)
        sorted_ids = class_ids[sorted_idx]

        class_buckets = [0, 125, 250, 500, 1000]
        num_buckets = len(class_buckets)-1
        acc_split_weights = np.vstack([np.ones((1, self.num_classes), dtype=np.float32), \
                                   np.zeros((num_buckets+1, self.num_classes), dtype=np.float32)])
        for i in range(num_buckets):
            s_idx = class_buckets[i]
            e_idx = min(class_buckets[i+1], self.num_classes)
            if s_idx > self.num_classes:
                break
            cur_bucket = set(sorted_ids[s_idx:e_idx])
            for l in range(self.num_classes):
                if l in cur_bucket:
                    acc_split_weights[i+1, l] = 1.0

        caffe_class_ids = set(caffe_class_ids)
        for l in range(self.num_classes):
            if l in caffe_class_ids:
                acc_split_weights[num_buckets+1, l] = 1.0

        return acc_split_weights


