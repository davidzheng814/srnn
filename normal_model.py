import logging
import tensorflow as tf
import glob

from tensorflow.python.ops import rnn_cell

import os
import random
import numpy as np
import argparse
from collections import namedtuple

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

NUM_EPOCHS = 400
BATCH_SIZE = 50
MAX_TIME = 100
MAX_FILES = None

LEARNING_RATE = 1e-3
BETA_1 = 0.9

USE_CHECKPOINT = False
CHECKPOINT = 'weights/model.ckpt'
LOGS_DIR = 'logs/'

DATA_DIR = 'numpy_arrays2/'
TRAIN_IND = 400
START_ITER = 0

INP_SIZE = 96
OUT_SIZE = 96

def pad(x):
    x = x[:,:MAX_TIME]
    r = np.zeros([x.shape[0], MAX_TIME, x.shape[2]])
    r[:x.shape[0],:x.shape[1],:x.shape[2]] = x

    return r
    
class Loader(object):
    def __init__(self, file_list):
        self.file_list = file_list
        self.load_dataset()

    def load_dataset(self):
        """"""
        self.inputs_train = np.zeros([0, MAX_TIME, INP_SIZE])
        self.inputs_val = np.zeros([0, MAX_TIME, INP_SIZE])
        self.outputs_train = np.zeros([0, MAX_TIME, OUT_SIZE])
        self.outputs_val = np.zeros([0, MAX_TIME, OUT_SIZE])

        for filename in self.file_list:
            data = np.load(filename)
            val_ind = int(data['val_index'])

            self.inputs_train = \
                np.append(self.inputs_train,
                    pad(data['inp'][:val_ind]), axis=0)

            self.inputs_val = \
                np.append(self.inputs_val,
                        pad(data['inp'][val_ind:]), axis=0)

            self.outputs_train = \
                np.append(self.outputs_train,
                        pad(data['out'][:val_ind]), axis=0)

            self.outputs_val = \
                np.append(self.outputs_val,
                        pad(data['out'][val_ind:]), axis=0)

        logging.info("Loaded %d files." % (len(self.file_list),))
        logging.info("Inputs Train Size: %d" %
                     (len(self.inputs_train),))
        logging.info("Inputs Val Size: %d" % (len(self.inputs_val),))

    def get_batches(self, is_train):
        if is_train:
            size = len(self.inputs_train)
            inds = range(0, size - BATCH_SIZE, BATCH_SIZE)
            random.shuffle(inds)
            data = [self.inputs_train, self.outputs_train]
        else:
            size = len(self.inputs_val)
            inds = range(0, size - BATCH_SIZE, BATCH_SIZE)
            random.shuffle(inds)
            data = [self.inputs_val, self.outputs_val]

        for ind in inds:
            end = ind + BATCH_SIZE
            yield [x[ind:end] for x in data]

def lstm(inputs, num_units):
    """
    Input: List[Tensor(batch_size, input_size)]
    Output: List[Tensor(batch_size, num_units)]
    """
    lstm_cell = rnn_cell.LSTMCell(num_units)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs,
            scope=tf.get_variable_scope(), dtype=tf.float32)

    return outputs

def dense(inp, output_size=1024):
    inp_shape = [int(x) for x in inp.get_shape()]
    inp_size = inp_shape[-1]
    h = tf.reshape(inp, [-1, inp_size])

    w = tf.get_variable("w", [inp_size, output_size],
        initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [output_size],
        initializer=tf.constant_initializer(0.0))

    h = tf.matmul(h, w) + b

    h = tf.reshape(h, inp_shape[:-1] + [-1])

    return h

def RNN(inputs, output_size=10):
    with tf.variable_scope('lstm1'):
        h = lstm(inputs, 512)

    with tf.variable_scope('lstm2'):
        h = lstm(h, 256)

    with tf.variable_scope('lstm3'):
        h = lstm(h, 256)

    with tf.variable_scope('dense1'):
        h = dense(h, output_size)

    return h

class SRNN(object):
    def __init__(self, sess, loader):
        self.sess = sess
        self.loader = loader

    def build_model(self):
        self.inp = tf.placeholder(tf.float32, 
            [BATCH_SIZE, MAX_TIME, INP_SIZE])
        self.out = tf.placeholder(tf.float32, 
            [BATCH_SIZE, MAX_TIME, OUT_SIZE])

        self.RNN = RNN(self.inp, 96)

        true = tf.reshape(self.out, [-1, OUT_SIZE])
        pred = tf.reshape(self.RNN, [-1, OUT_SIZE])
        self.loss = tf.reduce_mean(tf.squared_difference(true, pred))
        self.optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)
        self.merged = tf.merge_all_summaries()

    def run_batches(self, is_train):
        count = 0
        loss_sum = 0
        funcs = [self.loss]
        if is_train:
            funcs.append(self.optim)

        for inp, out in self.loader.get_batches(is_train):
            res = self.sess.run(funcs, feed_dict={
                self.inp:inp,
                self.out:out
            })
            loss_sum += res[0]
            count += 1

        return loss_sum / count

    def train_model(self):
        saver = tf.train.Saver()

        with self.sess as sess:
            if USE_CHECKPOINT and os.path.isfile(CHECKPOINT):
                logging.info("Restoring saved parameters")
                saver.restore(sess, CHECKPOINT)
            else:
                logging.info("Initializing parameters")
                sess.run(tf.initialize_all_variables())

            for epoch in range(NUM_EPOCHS):
                logging.info("Begin Epoch %d" % (epoch,))
                logging.info("Begin Train")
                loss = self.run_batches(True)
                logging.info("Train Loss: %0.4f" % (loss,))

                logging.info("Begin Val")
                loss = self.run_batches(False)
                logging.info("Val Loss: %0.4f" % (loss,))

                logging.info("Save Checkpoint")
                saver.save(sess, CHECKPOINT)

def main():
    sess = tf.Session()
    file_list = glob.glob(DATA_DIR + '/*.npz')
    if MAX_FILES:
        file_list = file_list[:MAX_FILES]
    logging.info("Loading Dataset")
    loader = Loader(file_list)
    srnn = SRNN(sess, loader)
    logging.info("Building Model")
    srnn.build_model()
    logging.info("Training Model")
    srnn.train_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--use-ckpt', action="store_true")
    parser.add_argument('--max-files', type=int)
    parser.add_argument('--start-iter', type=int)

    args = parser.parse_args()
    if args.num_epochs:
        NUM_EPOCHS = args.num_epochs
    if args.batch_size:
        NUM_EPOCHS = args.batch_size
    if args.use_ckpt:
        USE_CHECKPOINT = True
    if args.max_files:
        MAX_FILES = args.max_files
    if args.start_iter:
        START_ITER = args.start_iter

    main()
