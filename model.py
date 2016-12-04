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

NUM_EPOCHS = 100
BATCH_SIZE = 10
MAX_TIME = 150
MAX_FILES = None

LEARNING_RATE = .05
BETA_1 = 0.9

USE_CHECKPOINT = False
CHECKPOINT = 'weights/model.ckpt'
LOGS_DIR = 'logs/'

DATA_DIR = 'numpy_arrays/'
TRAIN_RATIO = 0.75

Node = namedtuple('Node', ['name', 'inp_size', 'time_size', 'out_size'])
NODES = [
    Node('s', 18, 36, 18),
    Node('a', 24, 48, 24),
    Node('l', 15, 30, 15),
]

Edge = namedtuple('Edge', ['node1', 'node2', 'size'])
EDGES = [
    [
        Edge('s', 'a', 42),
        Edge('s', 'l', 33),
    ], [
        Edge('a', 's', 42),
        Edge('a', 'a', 48),
    ], [
        Edge('l', 's', 33),
        Edge('l', 'l', 30),
    ]
]

def pad(x):
    r = np.zeros([x.shape[0], MAX_TIME, x.shape[2]])
    r[:x.shape[0],:x.shape[1],:x.shape[2]] = x

    return r

def get_name(edge, sort=True):
    if edge.node1 > edge.node2 and sort:
        edge.node2 + '_' + edge.node1
    return edge.node1 + '_' + edge.node2

class Loader(object):
    def __init__(self, file_list):
        self.file_list = file_list
        self.load_dataset()

    def load_dataset(self):
        """"""
        self.node_inputs = []
        self.time_inputs = []
        self.node_outputs = []
        self.edge_inputs = []
        for i, node in enumerate(NODES):
            self.node_inputs.append(np.zeros([0, MAX_TIME, node.inp_size]))
            self.time_inputs.append(np.zeros([0, MAX_TIME, node.time_size]))
            self.node_outputs.append(np.zeros([0, MAX_TIME, node.out_size]))

            self.edge_inputs.append([])
            for edge in EDGES[i]:
                self.edge_inputs[-1].append(np.zeros([0, MAX_TIME, edge.size]))

        for filename in self.file_list:
            data = np.load(filename)

            for i, node in enumerate(NODES):
                self.node_inputs[i] = \
                    np.append(self.node_inputs[i], pad(data[node.name + '_inp']), 
                              axis=0)

                self.time_inputs[i] = \
                    np.append(self.time_inputs[i], pad(data[node.name + '_time']),
                              axis=0)
                    
                self.node_outputs[i] = \
                    np.append(self.node_outputs[i], pad(data[node.name + '_out']),
                              axis=0)

                for j, edge in enumerate(EDGES[i]):
                    self.edge_inputs[i][j] = \
                        np.append(self.edge_inputs[i][j], 
                        pad(data[get_name(edge, sort=False)]),
                        axis=0)

        logging.info("Loaded %d files." % (len(self.file_list),))
        logging.info("Node Inputs Size: %d" % (len(self.node_inputs[0]),))

    def get_batches(self, node_ind, is_train):
        size = len(self.node_inputs[node_ind])
        num_train = int(TRAIN_RATIO * size)
        inds = range(0, size, BATCH_SIZE)[:-1]
        if is_train:
            inds = range(0, num_train - BATCH_SIZE, BATCH_SIZE)
            random.shuffle(inds)
        else:
            inds = range(num_train, size - BATCH_SIZE, BATCH_SIZE)

        inputs = [
            self.node_inputs[node_ind],
            self.time_inputs[node_ind],
            self.node_outputs[node_ind]
        ] + self.edge_inputs[node_ind]

        for ind in inds:
            end = ind + BATCH_SIZE
            yield [x[ind:end] for x in inputs]

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
    with tf.variable_scope('dense1'):
        h = dense(inputs, 256)

    with tf.variable_scope('dense2'):
        h = dense(h, 256)

    with tf.variable_scope('lstm1'):
        h = lstm(h, 512)

    return h

def nodeRNN(inputs, output_size=10, skip=None):
    with tf.variable_scope('lstm1'):
        h = lstm(inputs, 512)

    with tf.variable_scope('dense1'):
        h = dense(h, 256)

    with tf.variable_scope('dense2'):
        h = dense(h, 100)

    with tf.variable_scope('dense3'):
        h = dense(h, output_size)

    if skip is not None:
        h = h + skip

    return h

class SRNN(object):
    def __init__(self, sess, loader):
        self.sess = sess
        self.loader = loader

    def _model(self):

        self.nodeRNNs = []
        self.edgeRNNs = []
        for i, node in enumerate(NODES):
            self.edgeRNNs.append([])
            for j, edge in enumerate(EDGES[i]):
                with tf.variable_scope(get_name(edge)):
                    self.edgeRNNs[-1].append(edgeRNN(self.edge_inputs[i][j]))

            inp = tf.concat(2, 
                [self.node_inputs[i], self.time_inputs[i]] + self.edgeRNNs[i])

            with tf.variable_scope(node.name):
                self.nodeRNNs.append(nodeRNN(inp, output_size=node.out_size,
                    skip=self.node_inputs[i]))

    def build_model(self):
        self.edge_inputs = []
        self.node_inputs = []
        self.time_inputs = []
        self.node_outputs = []
        for i, node in enumerate(NODES):
            self.edge_inputs.append([])
            for edge in EDGES[i]:
                self.edge_inputs[-1].append(tf.placeholder(tf.float32,
                    [BATCH_SIZE, MAX_TIME, edge.size], 
                    get_name(edge, sort=False) + '_inp'))

            self.node_inputs.append(tf.placeholder(tf.float32,
                [BATCH_SIZE, MAX_TIME, node.inp_size], node.name  + '_inp'))
            self.time_inputs.append(tf.placeholder(tf.float32,
                [BATCH_SIZE, MAX_TIME, node.time_size], node.name  + '_inp'))
            self.node_outputs.append(tf.placeholder(tf.float32,
                [BATCH_SIZE, MAX_TIME, node.out_size], node.name  + '_inp'))

        self._model()

        self.losses = []
        self.optims = []
        for i, node in enumerate(NODES):
            true = tf.reshape(self.node_outputs[i], [-1, node.out_size])
            pred = tf.reshape(self.nodeRNNs[i], [-1, node.out_size])
            loss = tf.reduce_mean(tf.squared_difference(true, pred))
            self.losses.append(loss)
            tf.scalar_summary(node.name + '_loss', loss)
            optim = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
            self.optims.append(optim)

        self.merged = tf.merge_all_summaries()

    def _get_feed_dicts(self, all_batches):
        skipped = [False] * len(all_batches)
        i = 0
        num_skipped = 0
        while True:
            if skipped[i]:
                num_skipped += 1
                if num_skipped == len(all_batches):
                    return
                continue

            num_skipped = 0
            batch = next(all_batches[i], False)
            if batch:
                inputs = [self.node_inputs[i], self.time_inputs[i],
                    self.node_outputs[i]] + self.edge_inputs[i]
                feed_dict = {x:y for x, y in zip(inputs, batch)}
                yield self.losses[i], self.optims[i], feed_dict
            else:
                skipped[i] = True

            i = (i + 1) % len(all_batches)

    def run_batches(self, is_train, start_ind, writer):
        all_batches = [self.loader.get_batches(i, is_train) for i in range(len(NODES))]
        ind = start_ind
        count = 0
        loss_sum = 0

        for loss, optim, feed_dict in self._get_feed_dicts(all_batches):
            funcs = [loss]
            if is_train:
                funcs.append(optim)

            res = self.sess.run(funcs, feed_dict=feed_dict)
            loss_sum += res[0]

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
                logging.info("Begin Train")
                loss, ind = self.run_batches(True, ind, train_writer)
                logging.info("Train Loss: %0.4f" % (loss,))

                logging.info("Begin Val")
                loss, ind = self.run_batches(False, ind, val_writer)
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

    args = parser.parse_args()
    if args.num_epochs:
        NUM_EPOCHS = args.num_epochs
    if args.batch_size:
        NUM_EPOCHS = args.batch_size
    if args.use_ckpt:
        USE_CHECKPOINT = True
    if args.max_files:
        MAX_FILES = args.max_files

    main()
