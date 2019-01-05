# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
from scipy.sparse import vstack
import pickle
from collections import Counter

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class Dataset(object):
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
        self.batch_size = batch_size

        # load the data from the pickle file
        self._load_from_pickle_file()

        # number of samples in the dataset
        self.data_size = self.labels.shape[0]
 
        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            self.shuffle()

        # Initialise the counter
        self.counter = 0


    def _load_from_pickle_file(self):
        self.img_contents, self.labels, _ = pickle.load(open(self.pickle_file, 'rb'))
        if type(self.img_contents) is list:
            self.img_contents = np.concatenate(self.img_contents, axis = 0)


    def _one_hot(self, labels, num_classes):
        one_hot_labels = np.zeros((labels.size, num_classes), dtype=np.float32)
        one_hot_labels[np.arange(labels.size), labels] = 1
        return one_hot_labels


    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self.img_contents)
        index = np.arange(self.img_contents.shape[0])
        np.random.set_state(rng_state)
        np.random.shuffle(index)
        self.labels = self.labels[index, :]


    def reset(self):
        self.counter = 0


    def next_batch(self):
        s_idx = self.counter
        e_idx = self.counter + self.batch_size
        if e_idx <= self.data_size:
            img_batch = self.img_contents[s_idx: e_idx, :, :, :]
            lbl_batch = self.labels[s_idx: e_idx, :].todense()
        else:
            e_idx = self.batch_size - s_idx
            img_batch = np.concatenate(self.img_contents[s_idx:, :, :, :], self.img_contents[:e_idx, :, :, :], axis = 0)
            lbl_batch = vstack(self.labels[s_idx:, :], self.labels[:e_idx, :]).todense()
        self.counter = e_idx
        return img_batch, lbl_batch


    def get_acc_split_weights(self, caffe_class_ids, max_classes):
        if max_classes > self.num_classes or max_classes == 0:
            max_classes = self.num_classes
        class_freqs = self.labels.sum(0)
        class_freqs = np.reshape(np.array(class_freqs), [-1])[:max_classes]
        sorted_ids = np.argsort(-class_freqs)

        class_buckets = [0, 125, 250, 500, 1000, 10000]
        num_buckets = len(class_buckets)-1
        acc_split_weights = np.vstack([np.ones((1, max_classes), dtype=np.float32), \
                                   np.zeros((num_buckets+1, max_classes), dtype=np.float32)])
        for i in range(num_buckets):
            s_idx = class_buckets[i]
            e_idx = min(class_buckets[i+1], max_classes)
            if s_idx > max_classes:
                break
            cur_bucket = set(list(sorted_ids[s_idx:e_idx]))
            for l in range(max_classes):
                if l in cur_bucket:
                    acc_split_weights[i+1, l] = 1.0

        caffe_class_ids = set(caffe_class_ids)
        for l in range(max_classes):
            if l in caffe_class_ids:
                acc_split_weights[num_buckets+1, l] = 1.0

        if max_classes < self.num_classes:
            rem = self.num_classes-max_classes
            acc_split_weights = np.hstack([acc_split_weights, 
                                           np.zeros((num_buckets+2, rem), dtype=np.float32)])

        return acc_split_weights


