# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, batch_size, num_classes, shuffle=True):
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
        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)
 
        # if preload is True, load the images
        self._load_images()

        # convert lists to TF tensor
        self._convert_to_tensors()

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


    def _convert_to_tensors(self):
        """Convert list to tensors"""
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        self.img_contents = tf.stack(self.img_contents, axis=0)


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

