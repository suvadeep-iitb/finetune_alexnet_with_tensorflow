import os, time

import cv2
import numpy as np
import tensorflow as tf

from alexnet import AlexNet


flags = tf.flags

flags.DEFINE_string("image_paths", None,
                    "File storing the paths of the images")
flags.DEFINE_string("dest_dir", None,
                    "Name of the destination directory")

FLAGS=flags.FLAGS

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

def main(_):
    if not FLAGS.image_paths:
        raise ValueError("Must set --image_paths")
    if not FLAGS.dest_dir:
        raise ValueError("Must set --dest_dir")

    image_paths_file = FLAGS.image_paths
    dest_dir = FLAGS.dest_dir
    num_classes = 1000
    emb_dim = 4096

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    kp = tf.placeholder(tf.float32)

    # Initialize model
    train_layers = []
    model = AlexNet(x, kp, num_classes, emb_dim, train_layers)

    image_paths = open(image_paths_file).readlines()
    image_paths = [l.split()[0] for l in image_paths]

    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)

        pool5_tensor = sess.graph.get_tensor_by_name('pool5:0')

        for i, path in  enumerate(image_paths):

            # Read the image
            img = cv2.imread(path)
            # Convert image to float32 and resize to (227x227)
            img = cv2.resize(img.astype(np.float32), (227,227))

            # Subtract the ImageNet mean
            img -= imagenet_mean

            # Reshape as needed to feed into model
            img = img.reshape((1,227,227,3))

            feature = sess.run(pool5_tensor, feed_dict={x: img, kp: 1.0})
            feature = np.squeeze(np.reshape(feature, [-1, 6*6*256]))

            file_name = path.split('/')[-1]
            file_name = file_name.replace('.jpg', '.npy')

            save_path = os.path.join(dest_dir, file_name)
            np.save(save_path, feature)

            if (i + 1) % 1000 == 0:
                print('Processed '+str(i+1)+' files')




if __name__ == "__main__":
    tf.app.run()

