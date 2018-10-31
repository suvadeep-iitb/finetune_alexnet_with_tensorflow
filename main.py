import os, time

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.data import Iterator
from collections import Counter
from caffe_class_ids import caffe_class_ids


flags = tf.flags

flags.DEFINE_string("train_paths", None,
                    "File storing the paths of the train images")
flags.DEFINE_string("val_paths", None,
                    "File storing the paths of the validation images")
flags.DEFINE_string("test_paths", None,
                    "File storing the paths of the test images")
flags.DEFINE_string("checkpoint_path", None,
                    "Directory to save the checkpoints")
flags.DEFINE_string("filewriter_path", None,
                    "Directory to save the summery files")
flags.DEFINE_integer("max_threads", 0,
                     "Maximum number of threads/cores to be used during training")
flags.DEFINE_integer("display_step", 1,
                     "Step of percision on train, validation and test will be computed and printed")

flags.DEFINE_integer("num_classes", 1000,
                     "The number of classes, the dimension of the last layer")
flags.DEFINE_integer("embedding_dim", 4096,
                     "Embedding dimensions, the dimension of the second to last layer")
flags.DEFINE_integer("num_train_layers", 1,
                     "The number of layers (from the last) to be fine tuned during train")

flags.DEFINE_string("loss_func", "logistic",
                    "Loss function to be used for optimization. Can be one of ['mse', 'logistic', 'l2hinge', 'softmax']")
flags.DEFINE_float("keep_prob", 1.0,
                   "Keep probability to be used for dropout")
flags.DEFINE_float("exp", 1.0,
                   "exponent to be used element-wise power operations on the logits")
flags.DEFINE_float("learning_rate", 0.01,
                   "Initial learning rate")
flags.DEFINE_integer("num_epochs", 10,
                     "Tolal number of epochs")
flags.DEFINE_integer("batch_size", 128,
                     "Batch size for training")

FLAGS=flags.FLAGS


class ResultStruct:
    def __init__(self):
        self.acc = 0.0
        self.acc_split_top10 = 0.0
        self.acc_split_top10_100 = 0.0
        self.acc_split_top100_1000 = 0.0
        self.acc_split_top1000_10000 = 0.0
        self.acc_caffe = 0.0

    def add(self, acc_array):
        self.acc += acc_array[0]
        self.acc_split_top10 += acc_array[1]
        self.acc_split_top10_100 += acc_array[2]
        self.acc_split_top100_1000 += acc_array[3]
        self.acc_split_top1000_10000 += acc_array[4]
        self.acc_caffe += acc_array[5]

    def scaler_div(self, div):
        self.div = 1.0 * div
        self.acc /= div
        self.acc_split_top10 /= div
        self.acc_split_top10_100 /= div
        self.acc_split_top100_1000 /= div
        self.acc_split_top1000_10000 /=div
        self.acc_caffe /= div

    def __repr__(self):
        acc = self.acc
        t10 = self.acc_split_top10
        t100 = self.acc_split_top10_100
        t1K = self.acc_split_top100_1000
        t10K = self.acc_split_top1000_10000
        tc = self.acc_caffe
        rep_str = "%.4f (%.4f, %.4f, %.4f, %.4f) / (%.4f, %.4f)" % (acc, t10, t100, t1K, t10K, tc, acc - tc)
        return rep_str


def get_acc_split_weights(train_paths, num_classes, caffe_class_ids):
    lines = open(train_paths).readlines()
    class_counter = Counter([int(l.split()[1]) for l in lines])
    class_ids = np.array(list(class_counter.keys()))
    class_freqs = np.array(list(class_counter.values()))
    sorted_idx = np.argsort(-class_freqs)
    sorted_ids = class_ids[sorted_idx]

    class_buckets = [0, 10, 100, 1000, 10000]
    num_buckets = len(class_buckets)-1
    acc_split_weights = np.vstack([np.ones((1, num_classes), dtype=np.float32), \
                                   np.zeros((num_buckets+1, num_classes), dtype=np.float32)])
    for i in range(num_buckets):
        s_idx = class_buckets[i]
        e_idx = min(class_buckets[i+1], num_classes)
        if s_idx > num_classes:
            break
        cur_bucket = set(sorted_ids[s_idx:e_idx])
        for l in range(num_classes):
            if l in cur_bucket:
                acc_split_weights[i+1, l] = 1.0

    caffe_class_ids = set(caffe_class_ids)
    for l in range(num_classes):
        if l in caffe_class_ids:
            acc_split_weights[num_buckets+1, l] = 1.0

    return acc_split_weights


def main(_):
    if not FLAGS.train_paths:
        raise ValueError("Must set --train_paths")
    if not FLAGS.val_paths:
        raise ValueError("Must set --val_paths")
    if not FLAGS.test_paths:
        raise ValueError("Must set --test_paths")

    train_paths = FLAGS.train_paths
    val_paths = FLAGS.val_paths
    test_paths = FLAGS.test_paths
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    num_epochs = FLAGS.num_epochs
    emb_dim = FLAGS.embedding_dim
    keep_prob = FLAGS.keep_prob
    exp = FLAGS.exp
    loss_func = FLAGS.loss_func
    learning_rate = FLAGS.learning_rate
    checkpoint_path = FLAGS.checkpoint_path
    filewriter_path = FLAGS.filewriter_path
    display_step = FLAGS.display_step
    max_threads = FLAGS.max_threads

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 6*6*256])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    kp = tf.placeholder(tf.float32)

    # Initialize model
    layer_names = ['fc6', 'fc7', 'fc8']
    train_layers = layer_names[-FLAGS.num_train_layers:]
    model = AlexNet(x, kp, num_classes, emb_dim, train_layers)

    # Link variable to model output
    score = model.fc8
    if exp != 1.0:
        sign = tf.sign(score)
        score = tf.multiply(sign, tf.pow(tf.abs(score), exp))

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("loss_func"):
        if loss_func == 'softmax':
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                           labels=y,
                                                           name='softmax_loss')
        elif loss_func == 'logistic':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                           logits=score,
                                                           name='logistic_loss')
        elif loss_func == 'mse':
            loss = tf.losses.mean_squared_error(labels=y,
                                                predictions=score,
                                                name='mse_loss')
        elif loss_func == 'l2hinge':
            loss = tf.losses.hinge_loss(labels=y,
                                        logits=score,
                                        reduction=tf.losses.Reduction.NONE,
                                        name='l2hinge_loss')
            loss = tf.square(loss)
        loss = tf.reduce_mean(loss)

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)
 
    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('train_loss', loss)

    # Ops for evaluation
    acc_split_weights = get_acc_split_weights(FLAGS.train_paths, num_classes, caffe_class_ids)
    acc_split_weights = tf.convert_to_tensor(acc_split_weights, tf.float32)
    with tf.name_scope('accuracy'):
        label_splits = tf.matmul(acc_split_weights, tf.transpose(y))

        # ops for top 1 accuracies and their splitting
        top1_correct_pred =  tf.cast(tf.nn.in_top_k(score, tf.argmax(y, 1), 1), tf.float32)
        top1_correct_pred = tf.reshape(top1_correct_pred, [-1, 1])
        top1_accuracies = tf.squeeze(tf.matmul(label_splits, top1_correct_pred))/batch_size

        # ops for top 1 accuracies and their splitting
        top5_correct_pred =  tf.cast(tf.nn.in_top_k(score, tf.argmax(y, 1), 5), tf.float32)
        top5_correct_pred = tf.reshape(top5_correct_pred, [-1, 1])
        top5_accuracies = tf.squeeze(tf.matmul(label_splits, top5_correct_pred))/batch_size

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(train_paths,
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
        val_data = ImageDataGenerator(val_paths,
                                      batch_size=batch_size,
                                      num_classes=num_classes,
                                      shuffle=False)
        te_data = ImageDataGenerator(test_paths,
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=False)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    validation_init_op = iterator.make_initializer(val_data.data)
    testing_init_op = iterator.make_initializer(te_data.data)

    # Get the number of training/validation steps per epoch
    tr_batches_per_epoch = int(np.floor(tr_data.data_size/batch_size))
    val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
    te_batches_per_epoch = int(np.floor(te_data.data_size / batch_size))

    # Start Tensorflow session
    sess_conf = tf.ConfigProto(intra_op_parallelism_threads=max_threads, 
                               inter_op_parallelism_threads=max_threads)
    with tf.Session(config=sess_conf) as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))


        # Loop over number of epochs
        prev_top5_acc = 0
        counter = 0
        for epoch in range(num_epochs):

            # Initialize iterator with the training dataset
            start_time = time.time()
            sess.run(training_init_op)
            print('Train data init time: %.2f' % (time.time() - start_time))

            cost = 0.0
            load_time = 0
            train_time = 0
            for step in range(tr_batches_per_epoch):

                # get next batch of data
                start_time = time.time()
                img_batch, label_batch = sess.run(next_batch)
                load_time += time.time() - start_time

                # And run the training op
                start_time = time.time()
                _, lss = sess.run((train_op, loss), feed_dict={x: img_batch,
                                                               y: label_batch,
                                                               kp: keep_prob})
                cost += lss
                train_time += time.time() - start_time

                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            kp: 1.0})

                    writer.add_summary(s, epoch*tr_batches_per_epoch + step)

            elapsed_time = load_time + train_time
            print('Epoch: %d\tCost: %.6f\tElapsed Time: %.2f (%.2f / %.2f)' %
                    (epoch+1, cost/tr_batches_per_epoch, elapsed_time, load_time, train_time))

            # Test the model on the sampled train set
            tr_top1 = ResultStruct()
            tr_top5 = ResultStruct()
            # Evaluate on a for a smaller number of batches of trainset
            start_time = time.time()
            sess.run(training_init_op)
            print('Train (testing) data init time: %.2f' % (time.time() - start_time))
            start_time = time.time()
            num_batches = int(tr_batches_per_epoch/4);
            for _ in range(num_batches):

                img_batch, label_batch = sess.run(next_batch)
                temp_top1, temp_top5 = sess.run((top1_accuracies, top5_accuracies), 
                                                 feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            kp: 1.0});
                tr_top1.add(temp_top1)
                tr_top5.add(temp_top5)
            tr_top1.scaler_div(num_batches)
            tr_top5.scaler_div(num_batches)
            print('Epoch: ' + str(epoch+1) + '\tTrain Top 1 Acc: ' + str(tr_top1))
            print('Epoch: ' + str(epoch+1) + '\tTrain Top 5 Acc: ' + str(tr_top5))
            tr_pred_time = time.time() - start_time

            # Test the model on the entire validation set
            val_top1 = ResultStruct()
            val_top5 = ResultStruct()
            start_time = time.time()
            sess.run(validation_init_op)
            print('Validation data init time: %.2f' % (time.time() - start_time))
            start_time = time.time()
            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)
                temp_top1, temp_top5 = sess.run((top1_accuracies, top5_accuracies), 
                                                 feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            kp: 1.0})
                val_top1.add(temp_top1)
                val_top5.add(temp_top5)
            val_top1.scaler_div(val_batches_per_epoch)
            val_top5.scaler_div(val_batches_per_epoch)
            print('Epoch: ' + str(epoch+1) + '\tVal   Top 1 Acc: ' + str(val_top1))
            print('Epoch: ' + str(epoch+1) + '\tVal   Top 5 Acc: ' + str(val_top5))

            val_pred_time = time.time() - start_time

            # Test the model on the entire test set
            te_top1 = ResultStruct()
            te_top5 = ResultStruct()
            start_time = time.time()
            sess.run(testing_init_op)
            print('Test data init time: %.2f' % (time.time() - start_time))
            start_time = time.time()
            for _ in range(te_batches_per_epoch):

                img_batch, label_batch = sess.run(next_batch)
                temp_top1, temp_top5 = sess.run((top1_accuracies, top5_accuracies), 
                                                 feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            kp: 1.0});
                te_top1.add(temp_top1)
                te_top5.add(temp_top5)
            te_top1.scaler_div(te_batches_per_epoch)
            te_top5.scaler_div(te_batches_per_epoch)
            print('Epoch: ' + str(epoch+1) + '\tTest  Top 1 Acc: ' + str(te_top1))
            print('Epoch: ' + str(epoch+1) + '\tTest  Top 5 Acc: ' + str(te_top5))
            te_pred_time = time.time() - start_time

            elapsed_time = tr_pred_time + val_pred_time + te_pred_time
            print('Epoch %d Prediction: \tElapsed Time: %.2f (%.2f / %.2f / %.2f)'
                    % (epoch+1, elapsed_time, tr_pred_time, val_pred_time, te_pred_time))

            cur_top5_acc = val_top5.acc
            if cur_top5_acc - prev_top5_acc > 0.003:
                counter = 0
                prev_top5_acc = cur_top5_acc
            elif (cur_top5_acc - prev_top5_acc < -0.05) or (counter == 8):
                break
            else:
                counter += 1

            # save checkpoint of the model
            print("{} Saving checkpoint of model...".format(datetime.now()))
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))


if __name__ == "__main__":
    tf.app.run()

