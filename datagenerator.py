# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np

from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, preload=False, 
                 shuffle=True, buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
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
 
        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # if preload is True, load the images
        self.img_loaded = False
        if preload:
            self._load_images()
            self.img_loaded = True

        # convert lists to TF tensor
        self._convert_to_tensors()
     
        # create dataset
        if self.img_loaded:
            data = Dataset.from_tensor_slices((self.img_contents, self.labels))
        else:
            data = Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=10)

        elif mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=10)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data


    def _convert_to_tensors(self):
        """Convert list to tensors"""
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        if self.img_loaded:
            self.img_contents = tf.stack(self.img_contents, axis=0)


    def _load_images(self):
        """Load images into memory."""
        img_list = []
        for i in range(self.data_size):
            img_string = tf.read_file(tf.constant(self.img_paths[i]))
            img_decoded = tf.image.decode_jpeg(img_string, channels=3)
            img_resized = tf.image.resize_images(img_decoded, [227, 227])
            img_list.append(img_resized)
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


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        paths = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(paths[i])
            self.labels.append(labels[i])


    def _parse_function_train(self, img, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        if self.img_loaded:
            img_resized = img
        else:
            img_string = tf.read_file(img)
            img_decoded = tf.image.decode_jpeg(img_string, channels=3)
            img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot

    def _parse_function_inference(self, img, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        # load and preprocess the image
        if self.img_loaded:
            img_resized = img
        else:
            img_string = tf.read_file(img)
            img_decoded = tf.image.decode_jpeg(img_string, channels=3)
            img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot
