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

NUM_EPOCHS = 10
BATCH_SIZE = 5
MAX_TIME = 20

LEARNING_RATE = 0.001
BETA_1 = 0.9

USE_CHECKPOINT = False
CHECKPOINT = ''
LOGS_DIR = 'logs/'

DATA_DIR = 'data/'
TRAIN_RATIO = 0.75

HT_INP_SIZE = 5
OT_INP_SIZE = 5
HO_INP_SIZE = 5
OO_INP_SIZE = 5
H_INP_SIZE = 5
O_INP_SIZE = 5
H_OUT_SIZE = 5
O_OUT_SIZE = 5

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
            self.ht_inputs = np.append(self.ht_inputs, data['ht_inputs'], axis=0)
            self.ot_inputs = np.append(self.ot_inputs, data['ot_inputs'], axis=0)
            self.ho_inputs = np.append(self.ho_inputs, data['ho_inputs'], axis=0)
            self.oh_inputs = np.append(self.oh_inputs, data['oh_inputs'], axis=0)
            self.oo_inputs = np.append(self.oo_inputs, data['oo_inputs'], axis=0)
            self.h_inputs = np.append(self.h_inputs, data['h_inputs'], axis=0)
            self.o_inputs = np.append(self.o_inputs, data['o_inputs'], axis=0)
            self.h_outputs = np.append(self.h_outputs, data['h_outputs'], axis=0)
            self.o_outputs = np.append(self.o_outputs, data['o_outputs'], axis=0)

        logging.info("Loaded %d files." % (len(self.file_list),))

    def get_batches(self, is_h, is_train):
        size = len(self.h_outputs) if is_h else len(self.o_outputs)
        num_train = int(TRAIN_RATIO * size)
        inds = range(0, size, BATCH_SIZE)
        if is_train:
            inds = inds[:num_train]
        else:
            inds = inds[num_train:]

        random.shuffle(inds)
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
        self.ht_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, HT_INP_SIZE])
        self.ot_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, OT_INP_SIZE])
        self.ho_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, HO_INP_SIZE])
        self.oo_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, OO_INP_SIZE])
        self.h_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, H_INP_SIZE])
        self.o_inputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, O_INP_SIZE])
        self.h_outputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, H_OUT_SIZE])
        self.o_outputs = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_TIME, O_OUT_SIZE])

        self._model()
 
        h_out_re = tf.reshape(self.h_outputs, [-1, H_OUT_SIZE])
        H_re = tf.reshape(self.H, [-1, H_OUT_SIZE])
        o_out_re = tf.reshape(self.o_outputs, [-1, O_OUT_SIZE])
        O_re = tf.reshape(self.O, [-1, O_OUT_SIZE])

        self.h_loss = tf.nn.softmax_cross_entropy_with_logits(H_re, h_out_re)
        self.h_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1).minimize(self.h_loss)
        self.o_loss = tf.nn.softmax_cross_entropy_with_logits(O_re, o_out_re)
        self.o_optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1).minimize(self.o_loss)

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
            loss = self.h_loss
            optim = self.h_optim
        else:
            feed_dict = lambda x: {
                self.oo_inputs: x[0],
                self.ho_inputs: x[1],
                self.ot_inputs: x[2],
                self.o_inputs: x[3],
                self.o_outputs: x[4]
            }
            loss = self.o_loss
            optim = self.o_optim

        funcs = [loss, self.merged]
        if is_train:
            funcs.append(optim)

        for batch in batches:
            res = sess.run(funcs, feed_dict=feed_dict)
            loss_sum += res[0]
            writer.add_summary(res[1], ind)
            ind += 1
            count += 1

        return loss_sum / count, ind

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
                loss, ind = self.run_batches(True, True, ind, train_writer)
                logging.info("H Train Loss: %0.4f" % (loss,))

                logging.info("Begin O Train")
                loss, ind = self.run_batches(False, True, ind, train_writer)
                logging.info("O Train Loss: %0.4f" % (loss,))

                logging.info("Begin H Val")
                loss, ind = self.run_batches(True, False, ind, val_writer)
                logging.info("H Val Loss: %0.4f" % (loss,))

                logging.info("Begin O Val")
                loss, ind = self.run_batches(False, False, ind, val_writer)
                logging.info("H Val Loss: %0.4f" % (loss,))

                logging.info("Save Checkpoint")
                saver.save(sess, CHECKPOINT)

def main():
    sess = tf.Session()
    file_list = glob.glob(DATA_DIR)
    logging.info("Loading Dataset")
    loader = Loader(file_list)
    srnn = SRNN(sess, loader)
    logging.info("Building Model")
    srnn.build_model()
    logging.info("Training Model")
    srnn.train_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
