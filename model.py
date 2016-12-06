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
BATCH_SIZE = 100
MAX_TIME = 100
MAX_FILES = None

LEARNING_RATE = 1e-3
BETA_1 = 0.9

USE_CHECKPOINT = False
CHECKPOINT = 'weights/model.ckpt'
LOGS_DIR = 'logs/'

DATA_DIR = 'numpy_arrays/'
TRAIN_IND = 400
START_ITER = 0

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
    x = x[:,:MAX_TIME]
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
        self.node_inputs_train = []
        self.node_inputs_val = []
        self.time_inputs_train = []
        self.time_inputs_val = []
        self.node_outputs_train = []
        self.node_outputs_val = []
        self.edge_inputs_train = []
        self.edge_inputs_val = []
        for i, node in enumerate(NODES):
            self.node_inputs_train.append(np.zeros([0, MAX_TIME, node.inp_size]))
            self.node_inputs_val.append(np.zeros([0, MAX_TIME, node.inp_size]))
            self.time_inputs_train.append(np.zeros([0, MAX_TIME, node.time_size]))
            self.time_inputs_val.append(np.zeros([0, MAX_TIME, node.time_size]))
            self.node_outputs_train.append(np.zeros([0, MAX_TIME, node.out_size]))
            self.node_outputs_val.append(np.zeros([0, MAX_TIME, node.out_size]))

            self.edge_inputs_train.append([])
            self.edge_inputs_val.append([])
            for edge in EDGES[i]:
                self.edge_inputs_train[-1].append(np.zeros([0, MAX_TIME, edge.size]))
                self.edge_inputs_val[-1].append(np.zeros([0, MAX_TIME, edge.size]))

        for filename in self.file_list:
            data = np.load(filename)
            val_ind_ = int(data['val_index'])

            for i, node in enumerate(NODES):
                val_ind = val_ind_ if node.name == 's' else val_ind_ * 2
                self.node_inputs_train[i] = \
                    np.append(self.node_inputs_train[i],
                        pad(data[node.name + '_inp'][:val_ind]), axis=0)

                self.node_inputs_val[i] = \
                    np.append(self.node_inputs_val[i],
                        pad(data[node.name + '_inp'][val_ind:]), axis=0)

                self.time_inputs_train[i] = \
                    np.append(self.time_inputs_train[i],
                        pad(data[node.name + '_time'][:val_ind]), axis=0)
                    
                self.time_inputs_val[i] = \
                    np.append(self.time_inputs_val[i],
                        pad(data[node.name + '_time'][val_ind:]), axis=0)

                self.node_outputs_train[i] = \
                    np.append(self.node_outputs_train[i],
                        pad(data[node.name + '_out'][:val_ind]), axis=0)

                self.node_outputs_val[i] = \
                    np.append(self.node_outputs_val[i],
                        pad(data[node.name + '_out'][val_ind:]), axis=0)

                for j, edge in enumerate(EDGES[i]):
                    self.edge_inputs_train[i][j] = \
                        np.append(self.edge_inputs_train[i][j], 
                        pad(data[get_name(edge, sort=False)][:val_ind]),
                        axis=0)

                    self.edge_inputs_val[i][j] = \
                        np.append(self.edge_inputs_val[i][j], 
                        pad(data[get_name(edge, sort=False)][val_ind:]),
                        axis=0)

        logging.info("Loaded %d files." % (len(self.file_list),))
        logging.info("Node Inputs Train Size: %d" %
                     (len(self.node_inputs_train[0]),))
        logging.info("Node Inputs Val Size: %d" % (len(self.node_inputs_val[0]),))

    def get_batches(self, node_ind, is_train):
        if is_train:
            size = len(self.node_inputs_train[node_ind])
            inds = range(0, size - BATCH_SIZE, BATCH_SIZE)
            random.shuffle(inds)
            inputs = [
                self.node_inputs_train[node_ind],
                self.time_inputs_train[node_ind],
                self.node_outputs_train[node_ind]
            ] + self.edge_inputs_train[node_ind]
        else:
            size = len(self.node_inputs_val[node_ind])
            inds = range(0, size - BATCH_SIZE, BATCH_SIZE)
            inputs = [
                self.node_inputs_val[node_ind],
                self.time_inputs_val[node_ind],
                self.node_outputs_val[node_ind]
            ] + self.edge_inputs_val[node_ind]


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

def edgeRNN(inputs, noise=None):
    if noise is not None:
        inputs = inputs + tf.random_normal(inputs.get_shape(), stddev=noise)

    with tf.variable_scope('dense1'):
        h = dense(inputs, 256)

    with tf.variable_scope('dense2'):
        h = dense(h, 256)

    with tf.variable_scope('lstm1'):
        h = lstm(h, 512)

    return h

def nodeRNN(inputs, output_size=10, skip=None, noise=None):
    if noise is not None:
        inputs = inputs + tf.random_normal(inputs.get_shape(), stddev=noise)

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

        self.noise_vals = {
            109: 0.01,
            500: 0.05,
            1000: 0.1,
            1300: 0.2,
            2000: 0.3,
            2500: 0.5,
            3300: 0.7
        }

    def _model(self):
        self.nodeRNNs = []
        self.edgeRNNs = []
        for i, node in enumerate(NODES):
            self.edgeRNNs.append([])
            for j, edge in enumerate(EDGES[i]):
                with tf.variable_scope(get_name(edge)):
                    self.edgeRNNs[-1].append(edgeRNN(self.edge_inputs[i][j],
                        noise=self.noise_inp))

            inp = tf.concat(2, 
                [self.node_inputs[i], self.time_inputs[i]] + self.edgeRNNs[i])

            with tf.variable_scope(node.name):
                self.nodeRNNs.append(nodeRNN(inp, output_size=node.out_size,
                    skip=self.node_inputs[i], noise=self.noise_inp))

    def build_model(self):
        self.edge_inputs = []
        self.node_inputs = []
        self.time_inputs = []
        self.node_outputs = []
        self.noise_inp = tf.placeholder(tf.float32, [])
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
            optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
            # opt_func = tf.train.GradientDescentOptimizer(LEARNING_RATE)
            # gvs = opt_func.compute_gradients(loss)
            # gvs = [x for x in gvs if x[0] is not None]
            # gvs = [(tf.clip_by_value(grad, -5., 5.), var)
            #     for grad, var in gvs]
            # gvs = [(tf.clip_by_norm(grad, 25.), var)
            #     for grad, var in gvs]
            # optim = opt_func.apply_gradients(gvs)
            self.optims.append(optim)

        self.merged = tf.merge_all_summaries()

    def _get_feed_dicts(self, all_batches, is_train):
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
                if is_train and i == 0:
                    logging.info("Iter: %d" % (self.iter,))
                    if self.iter in self.noise_vals:
                        self.noise = self.noise_vals[self.iter]
                        logging.info("Adding Noise: %0.3f", self.noise)
                    self.iter += 1

                inputs = [self.node_inputs[i], self.time_inputs[i],
                    self.node_outputs[i]] + self.edge_inputs[i]
                feed_dict = {x:y for x, y in zip(inputs, batch)}
                feed_dict[self.noise_inp] = self.noise
                yield self.losses[i], self.optims[i], feed_dict
            else:
                skipped[i] = True

            i = (i + 1) % len(all_batches)

    def run_batches(self, is_train, writer):
        all_batches = [self.loader.get_batches(i, is_train) 
                       for i in range(len(NODES))]
        count = 0
        loss_sum = 0

        for loss, optim, feed_dict in self._get_feed_dicts(all_batches, is_train):
            funcs = [loss]
            if is_train:
                funcs.append(optim)

            res = self.sess.run(funcs, feed_dict=feed_dict)
            loss_sum += res[0]

            self.ind += 1
            count += 1

        return loss_sum / count

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

            self.ind, self.iter = 0, START_ITER
            self.noise = 0.
            for epoch in range(NUM_EPOCHS):
                logging.info("Begin Epoch %d" % (epoch,))
                logging.info("Begin Train")
                loss = self.run_batches(True, train_writer)
                logging.info("Train Loss: %0.4f" % (loss,))

                logging.info("Begin Val")
                loss = self.run_batches(False, val_writer)
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
