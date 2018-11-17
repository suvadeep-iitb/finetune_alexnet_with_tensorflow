import os, time
import pickle
import math as m

import cv2
import numpy as np
import tensorflow as tf

from alexnet import AlexNet


flags = tf.flags

flags.DEFINE_string("image_paths", None,
                    "File storing the paths of the images")
flags.DEFINE_string("save_file", None,
                    "Name of the destination directory")
flags.DEFINE_integer("batch_size", 1,
                     "Batch size")

FLAGS=flags.FLAGS

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

def main(_):
    if not FLAGS.image_paths:
        raise ValueError("Must set --image_paths")
    if not FLAGS.save_file:
        raise ValueError("Must set --save_file")

    image_paths_file = FLAGS.image_paths
    save_file = FLAGS.save_file
    batch_size = FLAGS.batch_size
    num_classes = 1000
    emb_dim = 4096

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    kp = tf.placeholder(tf.float32)

    # Initialize model
    train_layers = []
    model = AlexNet(x, kp, num_classes, emb_dim, train_layers)

    lines = open(image_paths_file).readlines()
    image_paths = [l.split()[0] for l in lines]
    def path2id(path):
        return path.split('/')[-1][:-4]
    ids = [path2id(p.split()[0]) for p in image_paths]
    labels = [int(l.split()[1]) for l in lines]

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)

        pool5_tensor = sess.graph.get_tensor_by_name('pool5:0')

        num_batches = m.ceil(len(image_paths)/float(batch_size))
        features = []
        for i  in  range(num_batches):

            s_batch = i * batch_size
            e_batch = min(s_batch+batch_size, len(image_paths))

            # Read the image
            img_pool = []
            for b in range(s_batch, e_batch):
                img = cv2.imread(image_paths[b])
                # Convert image to float32 and resize to (227x227)
                img = cv2.resize(img.astype(np.float32), (227,227))

                # Subtract the ImageNet mean
                img -= imagenet_mean

                # Reshape as needed to feed into model
                img_pool.append(img.reshape((227,227,3)))

            img_pool = np.stack(img_pool, axis = 0)

            feature = sess.run(pool5_tensor, feed_dict={x: img_pool, kp: 1.0})
            feature = np.squeeze(np.reshape(feature, [-1, 6*6*256]))
            features.append(feature)

            if (i + 1) % 10000 == 0:
                print('Processed '+str(i+1)+' files')

        features = np.vstack(features)

        batch_size = 50000
        array_list = []
        split_idx = np.arange(batch_size, len(image_paths), batch_size)
        features = np.split(features, split_idx, axis = 0)
        for i, feature in enumerate(features):
            print('Seg: %d size: %d' % (i, feature.shape[0]))

        pickle.dump((features, labels, ids), open(save_file, 'wb'))



if __name__ == "__main__":
    tf.app.run()

