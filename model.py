import logging
import tensorflow as tf
import glob

from tensorflow.python.ops import rnn_cell
from sklearn.metrics import f1_score

import os
import random
import numpy as np
import argparse
from collections import namedtuple

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

NUM_EPOCHS = 200
BATCH_SIZE = 5
MAX_TIME = 25

LEARNING_RATE = .05
BETA_1 = 0.9

USE_CHECKPOINT = False
CHECKPOINT = 'weights/model.ckpt'
LOGS_DIR = 'logs/'

DATA_DIR = 'numpy_arrays/'
TRAIN_RATIO = 0.75

HT_INP_SIZE = 160
OT_INP_SIZE = 40
HO_INP_SIZE = 400
OO_INP_SIZE = 400
H_INP_SIZE = 630
O_INP_SIZE = 180
H_OUT_SIZE = 10
O_OUT_SIZE = 12

def pad(x):
    r = np.zeros([x.shape[0], MAX_TIME, x.shape[2]])
    r[:x.shape[0],:x.shape[1],:x.shape[2]] = x

    return r

class Loader(object):
    def __init__(self, file_list):
        self.file_list = file_list
        self.load_dataset()

    def load_dataset(self):
        """
        Sets ht_inputs, ot_inputs, ho_inputs, oh_inputs, oo_inputs,
            h_inputs, o_inputs, h_outputs, o_outputs
        """
        self.ht_inputs = np.zeros([0, MAX_TIME, HT_INP_SIZE])
        self.ot_inputs = np.zeros([0, MAX_TIME, OT_INP_SIZE])
        self.ho_inputs = np.zeros([0, MAX_TIME, HO_INP_SIZE])
        self.oh_inputs = np.zeros([0, MAX_TIME, HO_INP_SIZE])
        self.oo_inputs = np.zeros([0, MAX_TIME, OO_INP_SIZE])
        self.h_inputs = np.zeros([0, MAX_TIME, H_INP_SIZE])
        self.o_inputs = np.zeros([0, MAX_TIME, O_INP_SIZE])
        self.h_outputs = np.zeros([0, MAX_TIME, H_OUT_SIZE])
        self.o_outputs = np.zeros([0, MAX_TIME, O_OUT_SIZE])
        for filename in self.file_list:
            data = np.load(filename)
            self.ht_inputs = np.append(self.ht_inputs, pad(data['ht_inputs']), axis=0)
            self.ot_inputs = np.append(self.ot_inputs, pad(data['ot_inputs']), axis=0)
            self.ho_inputs = np.append(self.ho_inputs, pad(data['ho_inputs']), axis=0)
            self.oh_inputs = np.append(self.oh_inputs, pad(data['oh_inputs']), axis=0)
            self.oo_inputs = np.append(self.oo_inputs, pad(data['oo_inputs']), axis=0)
            self.h_inputs = np.append(self.h_inputs, pad(data['h_inputs']), axis=0)
            self.o_inputs = np.append(self.o_inputs, pad(data['o_inputs']), axis=0)
            self.h_outputs = np.append(self.h_outputs, pad(data['h_outputs']), axis=0)
            self.o_outputs = np.append(self.o_outputs, pad(data['o_outputs']), axis=0)

        logging.info("Loaded %d files." % (len(self.file_list),))

    def get_batches(self, is_h, is_train):
        size = len(self.h_outputs) if is_h else len(self.o_outputs)
        num_train = int(TRAIN_RATIO * size)
        inds = range(0, size, BATCH_SIZE)[:-1]
        if is_train:
            inds = range(0, num_train - BATCH_SIZE, BATCH_SIZE)
            random.shuffle(inds)
        else:
            inds = range(num_train, size - BATCH_SIZE, BATCH_SIZE)

        if is_h:
            for ind in inds:
                end = ind + BATCH_SIZE
                yield (self.ho_inputs[ind:end], self.ht_inputs[ind:end],
                       self.h_inputs[ind:end], self.h_outputs[ind:end])
        else:
            for ind in inds:
                end = ind + BATCH_SIZE
                yield (self.oo_inputs[ind:end], self.oh_inputs[ind:end],
                       self.ot_inputs[ind:end], self.o_inputs[ind:end],
                       self.o_outputs[ind:end])

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

def edgeRNN(inputs):
    with tf.variable_scope('lstm1'):
        h = lstm(inputs, 128)

    return h

def nodeRNN(inputs, output_size=10):
    with tf.variable_scope('lstm1'):
        h = lstm(inputs, 256)

    with tf.variable_scope('dense1'):
        h = dense(h, output_size)

    return h

def scores(true, pred):
    rows = np.where(np.sum(true, 1) > 0)
    true = np.reshape(np.argmax(true[rows], 1), (-1,))
    pred = np.reshape(np.argmax(pred[rows], 1), (-1,))

    acc = np.mean(np.equal(true, pred))
    f1 = f1_score(true, pred, average='weighted')
    f1_micro = f1_score(true, pred, average='micro')
    f1_macro = f1_score(true, pred, average='macro')
    print f1, f1_micro, f1_macro
    return acc, f1

class SRNN(object):
    def __init__(self, sess, loader):
        self.sess = sess
        self.loader = loader

    def _model(self):
        with tf.variable_scope('ht'):
            ht_RNN = edgeRNN(self.ht_inputs)

        with tf.variable_scope('ot'):
            ot_RNN = edgeRNN(self.ot_inputs)

        with tf.variable_scope('ho'):
            ho_RNN = edgeRNN(self.ho_inputs)

        with tf.variable_scope('oo'):
            oo_RNN = edgeRNN(self.oo_inputs)

        with tf.variable_scope('H'):
            inp = tf.concat(2, [ho_RNN, ht_RNN, self.h_inputs])
            self.H = nodeRNN(inp, output_size=H_OUT_SIZE)

        with tf.variable_scope('O'):
            inp = tf.concat(2, [oo_RNN, ho_RNN, ot_RNN, self.o_inputs])
            self.O = nodeRNN(inp, output_size=O_OUT_SIZE)

    def build_model(self):
        self.ht_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, HT_INP_SIZE], 'ht_inp')
        self.ot_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, OT_INP_SIZE], 'ot_inp')
        self.ho_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, HO_INP_SIZE], 'ho_inp')
        self.oo_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, OO_INP_SIZE], 'oo_inp')
        self.h_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, H_INP_SIZE], 'h_inp')
        self.o_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, O_INP_SIZE], 'o_inp')
        self.h_outputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, H_OUT_SIZE], 'h_out')
        self.o_outputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, O_OUT_SIZE], 'o_out')

        self._model()
 
        self.h_true = tf.reshape(self.h_outputs, [-1, H_OUT_SIZE])
        self.h_pred = tf.reshape(self.H, [-1, H_OUT_SIZE])
        self.o_true = tf.reshape(self.o_outputs, [-1, O_OUT_SIZE])
        self.o_pred = tf.reshape(self.O, [-1, O_OUT_SIZE])

        self.h_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.h_pred, self.h_true))
        tf.scalar_summary('h_loss', self.h_loss)
        # self.h_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1).minimize(self.h_loss)
        self.h_optim = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.h_loss)
        self.o_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.o_pred, self.o_true))
        tf.scalar_summary('o_loss', self.o_loss)
        # self.o_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1).minimize(self.o_loss)
        self.o_optim = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.o_loss)

        self.merged = tf.merge_all_summaries()

    def run_batches(self, is_h, is_train, start_ind, writer):
        batches = self.loader.get_batches(is_h, is_train)
        ind = start_ind
        loss_sum = 0
        count = 0
        if is_h:
            feed_dict = lambda x: {
                self.ho_inputs: x[0], 
                self.ht_inputs: x[1],
                self.h_inputs: x[2],
                self.h_outputs: x[3]
            }
            loss, true, pred = self.h_loss, self.h_true, self.h_pred
            arr = [np.zeros((0, H_OUT_SIZE)), np.zeros((0, H_OUT_SIZE))]
            optim = self.h_optim
        else:
            feed_dict = lambda x: {
                self.oo_inputs: x[0],
                self.ho_inputs: x[1],
                self.ot_inputs: x[2],
                self.o_inputs: x[3],
                self.o_outputs: x[4]
            }
            loss, true, pred = self.o_loss, self.o_true, self.o_pred
            arr = [np.zeros((0, O_OUT_SIZE)), np.zeros((0, O_OUT_SIZE))]
            optim = self.o_optim

        funcs = [true, pred, loss]
        if is_train:
            funcs.append(optim)
        
        for batch in batches:
            res = self.sess.run(funcs, feed_dict=feed_dict(batch))
            arr = [np.append(arr[i], res[i], axis=0) for i in range(2)]
            loss_sum += res[2]
            ind += 1
            count += 1

        acc, f1 = scores(arr[0], arr[1])

        return loss_sum / count, ind, acc, f1

    def train_model(self):
        train_writer = tf.train.SummaryWriter(os.path.join(LOGS_DIR, 'train'),
                graph=self.sess.graph)
        val_writer = tf.train.SummaryWriter(os.path.join(LOGS_DIR, 'val'),
                graph=self.sess.graph)

        saver = tf.train.Saver()

        with self.sess as sess:
            if USE_CHECKPOINT and os.path.isfile(CHECKPOINT):
                logging.info("Restoring saved parameters")
                saver.restore(sess, CHECKPOINT)
            else:
                logging.info("Initializing parameters")
                sess.run(tf.initialize_all_variables())

            ind = 0
            for epoch in range(NUM_EPOCHS):
                logging.info("Begin Epoch %d" % (epoch,))
                logging.info("Begin H Train")
                loss, ind, acc, f1 = self.run_batches(True, True, ind, train_writer)
                logging.info("H Train Loss: %0.4f, Acc: %0.4f, F1: %0.4f" % (loss, acc, f1))

                logging.info("Begin O Train")
                loss, ind, acc, f1 = self.run_batches(False, True, ind, train_writer)
                logging.info("O Train Loss: %0.4f, Acc: %0.4f, F1: %0.4f" % (loss, acc, f1))

                logging.info("Begin H Val")
                loss, ind, acc, f1 = self.run_batches(True, False, ind, val_writer)
                logging.info("H Val Loss: %0.4f, Acc: %0.4f, F1: %0.4f" % (loss, acc, f1))

                logging.info("Begin O Val")
                loss, ind, acc, f1 = self.run_batches(False, False, ind, val_writer)
                logging.info("O Val Loss: %0.4f, Acc: %0.4f, F1: %0.4f" % (loss, acc, f1))

                logging.info("Save Checkpoint")
                saver.save(sess, CHECKPOINT)

def main():
    sess = tf.Session()
    file_list = glob.glob(DATA_DIR + '/*')
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

    args = parser.parse_args()
    if args.num_epochs:
        NUM_EPOCHS = args.num_epochs
    if args.batch_size:
        NUM_EPOCHS = args.batch_size
    if args.use_ckpt:
        USE_CHECKPOINT = True

    main()
