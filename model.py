import logging
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

import numpy as np
import argparse
from collections import namedtuple

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

BATCH_SIZE = 50
TIME_STEPS = 1
LEARNING_RATE = 0.001
BETA_1 = 0.9
EDGE_INPUT_SIZES = {
    'H-H': 10,
    'H-O': 5,
    'O-O': 15
}

NODE_INPUT_SIZES = {
    'H': 6,
    'O': 5,
}

NODE_OUTPUT_SIZES = {
    'H': 12,
    'O': 15,
}

NODES = [
    ('a', 'H'),
    ('b', 'O'),
    ('c', 'O'),
    ('d', 'O'),
]

EDGES = [
    ('a', 'b'),
    ('a', 'c'),
    ('a', 'd'),
]


class Loader(object):
    def __init__(self):
        pass

    def batch(self):
        """Returns train, val, test tuple."""
        pass

def lstm(inputs, num_units):
    """
    Input: List[Tensor(batch_size, input_size)]
    Output: List[Tensor(batch_size, num_units)]
    """
    lstm_cell = rnn_cell.LSTMCell(num_units)
    outputs, state = tf.nn.rnn(lstm_cell, inputs,
            scope=tf.get_variable_scope(), dtype=tf.float32)

    return outputs

def dense(inputs, output_size):
    """
    Input: List[Tensor(batch_size, input_size)]
    Output: List[Tensor(batch_size, output_size)]
    """
    inp_size = inputs[0].get_shape()
    h = [tf.reshape(x, [int(inp_size[0]), -1]) for x in inputs]
    h_size = h[0].get_shape()[1]

    w = tf.get_variable("w", [h_size, output_size],
            initializer=tf.random_normal_initializer(stddev=0.02))
    b = tf.get_variable("b", [output_size],
            initializer=tf.constant_initializer(0.0))

    h = [tf.matmul(x, w) + b for x in h]

    # TODO Skip input and output connections.

    return h

def edgeRNN(inputs):
    with tf.variable_scope('dense1'):
        h = dense(inputs, 256)

    with tf.variable_scope('dense2'):
        h = dense(h, 256)

    with tf.variable_scope('lstm1'):
        h = lstm(h, 512)

    return h

def nodeRNN(inputs, output_size=10):
    with tf.variable_scope('lstm1'):
        h = lstm(inputs, 512)

    with tf.variable_scope('dense1'):
        h = dense(h, 256)

    with tf.variable_scope('dense2'):
        h = dense(h, 100)

    with tf.variable_scope('dense3'):
        h = lstm(h, output_size)

    return h

def get_edge(x, y):
    if x < y:
        return x + '-' + y
    else:
        return y + '-' + x

def optimizer(y_, y):
    loss = tf.reduce_mean(tf.squared_difference(y_, y))
    optim = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA_1).minimize(loss)
    return optim

class SRNN(object):
    def __init__(self, sttypes, stneighbors):
        self.sttypes = sttypes
        self.stneighbors = stneighbors
        self.scopes = set()

    def build_model(self):
        self.edge_inputs = {}
        self.node_inputs = {}
        self.node_outputs = {}
        self.nodeRNNs = {}
        self.optimizers = {}

        for node, neighbors in self.stneighbors.items():
            neighbors = neighbors + [node]
            print "NODE:", node
            node_type = self.sttypes[node]
            edge_inputs = {}
            self.edge_inputs[node] = edge_inputs
            edge_type_inputs = {}
            for neighbor in neighbors:
                edge = get_edge(node, neighbor)
                neighbor_type = self.sttypes[neighbor]
                edge_type = get_edge(node_type, neighbor_type)

                edge_inputs[edge] = [tf.placeholder(tf.float32,
                        [BATCH_SIZE, EDGE_INPUT_SIZES[edge_type]]) for _ in range(TIME_STEPS)]

                if edge_type in edge_type_inputs:
                    edge_type_inputs[edge_type] = [tf.add(x, y)
                            for x, y in zip(edge_inputs[edge], edge_type_inputs[edge_type])]
                else:
                    edge_type_inputs[edge_type] = edge_inputs[edge]

            self.node_inputs[node] = [tf.placeholder(tf.float32,
                    [BATCH_SIZE, NODE_INPUT_SIZES[node_type]]) for _ in range(TIME_STEPS)]

            nodeRNN_input = self.node_inputs[node]
            for edge_type, edge_type_input in edge_type_inputs.items():
                print edge_type
                with tf.variable_scope(edge_type) as scope:
                    if edge_type in self.scopes:
                        scope.reuse_variables()
                    rnn = edgeRNN(edge_type_input)
                    nodeRNN_input = [tf.concat(1, [x, y])
                            for x, y in zip(nodeRNN_input, rnn)]
                self.scopes.add(edge_type)

            print node_type
            self.node_outputs[node] = [tf.placeholder(tf.float32,
                [BATCH_SIZE, NODE_OUTPUT_SIZES[node_type]]) for _ in range(TIME_STEPS)]
            with tf.variable_scope(node_type) as scope:
                if node_type in self.scopes:
                    scope.reuse_variables()
                self.nodeRNNs[node] = nodeRNN(nodeRNN_input,
                        output_size=NODE_OUTPUT_SIZES[node_type])
                self.optimizers[node] = optimizer(self.nodeRNNs[node], self.node_outputs[node])
            self.scopes.add(node_type)

    def _train(self):


    def train_model(self):
        tf.initialize_all_variables().run()
        merged_summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(LOGS_DIR, graph_def=self.sess.graph_def)
        for epoch in range(NUM_EPOCHS):
            pass

def main():
    sttypes = {
        'a': 'H',
        'b': 'O',
        'c': 'O'
    }

    stneighbors = {
        'a': ['b', 'c'],
        'b': ['a', 'c'],
        'c': ['a', 'b']
    }

    srnn = SRNN(sttypes, stneighbors)
    srnn.build_model()
    print srnn.scopes
    print srnn.edge_inputs
    print srnn.node_inputs
    print srnn.nodeRNNs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()
