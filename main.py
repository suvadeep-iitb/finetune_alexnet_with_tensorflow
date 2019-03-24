import os, time, math

import numpy as np
import tensorflow as tf

from alexnet import AlexNet
from dataset import Dataset
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
flags.DEFINE_string("corr_path", None,
                    "File containing label correlations")
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
                    "Loss function to be used for optimization. Can be one of \
                    ['mse', 'logistic', 'l2hinge', 'softmax', 'hinge', 'sigmoid', 'boot_soft', 'boot_hard', 'ramp']")
flags.DEFINE_float("keep_prob", 1.0,
                   "Keep probability to be used for dropout")
flags.DEFINE_float("exp", 1.0,
                   "exponent to be used element-wise power operations on the logits")
flags.DEFINE_float("c", 0.005,
                   "cutoff to be used in eexponentiation")
flags.DEFINE_integer("nel", 0,
                     "number of eexponentiation layer to be inserted")
flags.DEFINE_float("weight", 0.0,
                   "weight for the label correlations in loss function")
flags.DEFINE_float("learning_rate", 0.01,
                   "Initial learning rate")
flags.DEFINE_integer("num_epochs", 10,
                     "Tolal number of epochs")
flags.DEFINE_integer("batch_size", 128,
                     "Batch size for training")
flags.DEFINE_float("beta", 0.8,
                     "The value of beta to be used in boot_soft and boot_hard loss functions")

FLAGS=flags.FLAGS


class ResultStruct:
    def __init__(self):
        self.acc = 0.0
        self.acc_split_1 = 0.0
        self.acc_split_2 = 0.0
        self.acc_split_3 = 0.0
        self.acc_split_4 = 0.0
        self.acc_split_5 = 0.0
        self.acc_caffe = 0.0

    def add(self, acc_array):
        self.acc += acc_array[0]
        self.acc_split_1 += acc_array[1]
        self.acc_split_2 += acc_array[2]
        self.acc_split_3 += acc_array[3]
        self.acc_split_4 += acc_array[4]
        self.acc_split_5 += acc_array[5]
        self.acc_caffe += acc_array[6]

    def scaler_div(self, div):
        self.div = 1.0 * div
        self.acc /= div
        self.acc_split_1 /= div
        self.acc_split_2 /= div
        self.acc_split_3 /= div
        self.acc_split_4 /=div
        self.acc_split_5 /=div
        self.acc_caffe /= div

    def __repr__(self):
        acc = self.acc
        t1 = self.acc_split_1
        t2 = self.acc_split_2
        t3 = self.acc_split_3
        t4 = self.acc_split_4
        t5 = self.acc_split_5
        tc = self.acc_caffe
        rep_str = "%.4f (%.4f, %.4f, %.4f, %.4f, %.4f) / (%.4f, %.4f)" % (acc, t1, t2, t3, t4, t5, tc, acc - tc)
        return rep_str



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
    corr_path = FLAGS.corr_path
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    num_epochs = FLAGS.num_epochs
    emb_dim = FLAGS.embedding_dim
    keep_prob = FLAGS.keep_prob
    exp = FLAGS.exp
    c = FLAGS.c
    nel = FLAGS.nel
    weight = FLAGS.weight
    loss_func = FLAGS.loss_func
    learning_rate = FLAGS.learning_rate
    checkpoint_path = FLAGS.checkpoint_path
    filewriter_path = FLAGS.filewriter_path
    display_step = FLAGS.display_step
    max_threads = FLAGS.max_threads
    beta = FLAGS.beta

    # Create parent path if it doesn't exist
    '''
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)
    '''

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 13, 13, 256])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    kp = tf.placeholder(tf.float32)

    # Initialize model
    layer_names = ['conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    train_layers = layer_names[-FLAGS.num_train_layers:]
    model = AlexNet(x, kp, exp, num_classes, emb_dim, train_layers, c, nel)

    # Link variable to model output
    score = model.fc8

    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("loss_func"):
        if loss_func == 'softmax':
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                           labels=y,
                                                           name='softmax_loss_1')
        elif loss_func == 'logistic':
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                           logits=score,
                                                           name='logistic_loss_1')
        elif loss_func == 'mse':
            loss = tf.losses.mean_squared_error(labels=y,
                                                predictions=score)
        elif loss_func == 'l2hinge':
            loss = tf.losses.hinge_loss(labels=y,
                                        logits=score,
                                        reduction=tf.losses.Reduction.NONE)
            loss = tf.square(loss)
        elif loss_func == 'hinge':
            loss = tf.losses.hinge_loss(labels=y,
                                        logits=score,
                                        reduction=tf.losses.Reduction.NONE)
        elif loss_func == 'boot_soft':
            sm_score = tf.nn.softmax(score)
            bs_y = (beta * y + (1.0 - beta) * sm_score)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                           labels=bs_y)
        elif loss_func == 'boot_hard':
            pred_y = tf.one_hot(tf.argmax(score, 1), tf.shape(y)[1])
            bh_y = (beta * y + (1.0 - beta) * pred_y)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                           labels=bh_y)
        elif loss_func == 'sigmoid':
            beta = 1.0
            y_signed = 2.0*y - 1.0
            score = tf.layers.batch_normalization(score, axis=1)
            loss = tf.sigmoid(-beta*tf.multiply(y_signed, score))
        elif loss_func == 'ramp':
            beta = 1.0
            y_signed = 2.0*y - 1.0
            score = tf.layers.batch_normalization(score, axis=1)
            loss = tf.minimum(1.0, tf.maximum(0.0, 1.0 - beta * y_signed * score))
            
        loss = tf.reduce_mean(loss)
        absmean = tf.reduce_mean(tf.abs(y*score))
        mean, var = (tf.nn.moments(y*score, axes=[0]))

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)
 
    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('train_loss', loss)

    # Initialize the datasets
    start_time = time.time()
    tr_data = Dataset(train_paths, batch_size, num_classes, True)
    val_data = Dataset(val_paths, batch_size, num_classes, False)
    te_data = Dataset(test_paths, batch_size, num_classes, False)
    load_time = time.time() - start_time
    log_buff = 'Data loading time: %.2f' % load_time + '\n'

    # Ops for evaluation
    acc_split_weights = tr_data.get_acc_split_weights(caffe_class_ids, 0)
    acc_split_weights = tf.convert_to_tensor(acc_split_weights, tf.float32)
    with tf.name_scope('accuracy'):
        # ops for top 1 accuracies and their splitting
        top1_score, _ = tf.nn.top_k(score, 1)
        top1_thresolds = tf.reduce_min(top1_score, axis = 1, keepdims = True)
        top1_thresolds_bc = tf.broadcast_to(top1_thresolds, score.shape)
        top1_y = tf.to_float(tf.math.greater_equal(score, top1_thresolds_bc))
        top1_correct_pred = tf.multiply(y, top1_y)
        top1_precision = tf.squeeze(tf.reduce_mean(tf.matmul(acc_split_weights, \
                                                   tf.transpose(top1_correct_pred)), axis = 1))

        # ops for top 5 accuracies and their splitting
        top5_score, _ = tf.nn.top_k(score, min(5, num_classes))
        top5_thresolds = tf.reduce_min(top5_score, axis = 1, keepdims = True)
        top5_thresolds_bc = tf.broadcast_to(top5_thresolds, score.shape)
        top5_y = tf.to_float(tf.math.greater_equal(score, top5_thresolds_bc))
        top5_correct_pred = tf.multiply(y, top5_y)
        top5_precision = tf.squeeze(tf.reduce_mean(tf.matmul(acc_split_weights, \
                                                   tf.transpose(top5_correct_pred)), axis = 1))

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    #writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()


    # Start Tensorflow session
    sess_conf = tf.ConfigProto(intra_op_parallelism_threads=max_threads, 
                               inter_op_parallelism_threads=max_threads)
    with tf.Session(config=sess_conf) as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Add the model graph to TensorBoard
        #writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        #model.load_initial_weights(sess)

        log_buff += "{} Start training...".format(datetime.now())+'\n'
        #print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
        #                                              filewriter_path))


        print(log_buff)
        log_buff = ''

        # Loop over number of epochs
        prev_acc = 0
        counter = 0
        for epoch in range(num_epochs):

            log_buff += "{} Epoch: {}".format(datetime.now(), epoch)+'\n'

            tr_batches_per_epoch = tr_data.data_size // batch_size
            tr_data.reset()
            start_time = time.time()
            tr_data.shuffle()
            shuffle_time = time.time() - start_time
            log_buff += 'Train data shuffling time: %.2f' % shuffle_time + '\n'
            cost = 0.0
            load_time = 0
            train_time = 0

            tmn = 0.0
            tabsmn = 0.0
            tva = 0.0
            for step in range(tr_batches_per_epoch):

                # get next batch of data
                start_time = time.time()
                img_batch, label_batch = tr_data.next_batch()
                load_time += time.time() - start_time

                # And run the training op
                start_time = time.time()
                _, lss, mn, absmn, va = sess.run((train_op, loss, mean, absmean, var), feed_dict={x: img_batch,
                                                               y: label_batch,
                                                               kp: keep_prob})
                cost += lss
                train_time += time.time() - start_time

                tmn += np.mean(mn)
                tabsmn += absmn
                tva += np.mean(va)

                # Generate summary with the current batch of data and write to file
                '''
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            kp: 1.0})

                    writer.add_summary(s, epoch*tr_batches_per_epoch + step)
                '''

            elapsed_time = load_time + train_time
            cost /= tr_batches_per_epoch
            log_buff += 'Epoch: %d\tCost: %.6f\tElapsed Time: %.2f (%.2f / %.2f)' % \
                    (epoch+1, cost, elapsed_time, load_time, train_time) + '\n'

            tmn /= tr_batches_per_epoch
            tabsmn /= tr_batches_per_epoch
            tva /= tr_batches_per_epoch
            log_buff += 'Mean: %.4f\tAbsMean: %.4f\tSD: %.4f\n' % (tmn, tabsmn, np.sqrt(tva))

            # Test the model on the sampled train set
            tr_top1 = ResultStruct()
            tr_top5 = ResultStruct()
            # Evaluate on a for a smaller number of batches of trainset
            tr_data.reset()
            start_time = time.time()
            num_batches = int(tr_batches_per_epoch/4);
            for _ in range(num_batches):

                img_batch, label_batch = tr_data.next_batch()
                prec1, prec5 = sess.run((top1_precision, top5_precision), \
                                                 feed_dict={x: img_batch, \
                                                            y: label_batch, \
                                                            kp: 1.0});
                tr_top1.add(prec1)
                tr_top5.add(prec5)
            tr_top1.scaler_div(num_batches)
            tr_top5.scaler_div(num_batches)
            log_buff += 'Epoch: ' + str(epoch+1) + '\tTrain Top 1 Acc: ' + str(tr_top1) + '\n'
            log_buff += 'Epoch: ' + str(epoch+1) + '\tTrain Top 5 Acc: ' + str(tr_top5) + '\n'
            tr_pred_time = time.time() - start_time

            # Test the model on the entire validation set
            val_top1 = ResultStruct()
            val_top5 = ResultStruct()
            val_data.reset()
            val_batches_per_epoch = val_data.data_size // batch_size
            start_time = time.time()
            for _ in range(val_batches_per_epoch):

                img_batch, label_batch = val_data.next_batch()
                prec1, prec5 = sess.run((top1_precision, top5_precision), \
                                                 feed_dict={x: img_batch, \
                                                            y: label_batch, \
                                                            kp: 1.0})
                val_top1.add(prec1)
                val_top5.add(prec5)
            val_top1.scaler_div(val_batches_per_epoch)
            val_top5.scaler_div(val_batches_per_epoch)
            log_buff += 'Epoch: ' + str(epoch+1) + '\tVal   Top 1 Acc: ' + str(val_top1) + '\n'
            log_buff += 'Epoch: ' + str(epoch+1) + '\tVal   Top 5 Acc: ' + str(val_top5) + '\n'

            val_pred_time = time.time() - start_time

            # Test the model on the entire test set
            te_top1 = ResultStruct()
            te_top5 = ResultStruct()
            te_data.reset()
            te_batches_per_epoch = te_data.data_size // batch_size
            start_time = time.time()
            for _ in range(te_batches_per_epoch):

                img_batch, label_batch = te_data.next_batch()
                prec1, prec5 = sess.run((top1_precision, top5_precision), \
                                                 feed_dict={x: img_batch, \
                                                            y: label_batch, \
                                                            kp: 1.0});
                te_top1.add(prec1)
                te_top5.add(prec5)
            te_top1.scaler_div(te_batches_per_epoch)
            te_top5.scaler_div(te_batches_per_epoch)
            log_buff += 'Epoch: ' + str(epoch+1) + '\tTest  Top 1 Acc: ' + str(te_top1) + '\n'
            log_buff += 'Epoch: ' + str(epoch+1) + '\tTest  Top 5 Acc: ' + str(te_top5) + '\n'
            te_pred_time = time.time() - start_time

            elapsed_time = tr_pred_time + val_pred_time + te_pred_time
            log_buff += 'Epoch %d Prediction: \tElapsed Time: %.2f (%.2f / %.2f / %.2f)' \
                    % (epoch+1, elapsed_time, tr_pred_time, val_pred_time, te_pred_time) + '\n'

            if math.isnan(cost):
                break
            cur_acc = val_top5.acc
            if cur_acc - prev_acc > 0.003:
                counter = 0
                prev_acc = cur_acc
            elif (cur_acc - prev_acc < -0.5) or (counter == 15):
                break
            else:
                counter += 1

            # save checkpoint of the model
            '''
            print("{} Saving checkpoint of model...".format(datetime.now()))
            checkpoint_name = os.path.join(checkpoint_path,
                                           'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                           checkpoint_name))
            '''
           
            print(log_buff)
            log_buff = ''

        print(log_buff)


if __name__ == "__main__":
    tf.app.run()

