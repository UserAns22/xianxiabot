import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np

BATCH_SIZE = 32
SEQ_LENGTH = 64
RNN_SIZE = 128
NUM_LAYERS = 2
learning_rate = .01
decay_rate = .97
output_keep_prob = .97
input_keep_prob = .97
grad_clip = 5.
cell_fn = rnn.LSTMCell
training = True

"credits to https://github.com/sherjilozair/char-rnn-tensorflow/"

class Model():

    def __init__(self, vocab_size, training = True):

        cells = []
        for _ in range(NUM_LAYERS):
            cell = cell_fn(RNN_SIZE)
            if training and (output_keep_prob < 1.0 or input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                    input_keep_prob=input_keep_prob,
                    output_keep_prob= output_keep_prob)
            cells.append(cell)
                
        self.cell = rnn.MultiRNNCell(cells, state_is_tuple = True)
        self.input_data = tf.placeholder( tf.int32, [BATCH_SIZE, SEQ_LENGTH])
        self.targets = tf.placeholder( tf.int32, [BATCH_SIZE, SEQ_LENGTH])
        self.initial_state = cell.zero_state([BATCH_SIZE], tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",[RNN_SIZE, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

        embedding = tf.get_variable("embedding", [vocab_size, RNN_SIZE])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        if training and output_keep_prob:
            inputs = tf.nn.dropout(inputs, output_keep_prob)
        

        inputs = tf.split(inputs, SEQ_LENGTH, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        
        def loop(prev, _):
            prev = tf.matmul(prev,softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        if training:
           loop = None
        outputs, last_state =  legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, 
                loop_function = loop, scope = 'rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, RNN_SIZE])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([BATCH_SIZE * SEQ_LENGTH])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / BATCH_SIZE / SEQ_LENGTH
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable = False)
        tvars = tf.trainable_variables()
        grads = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)[0]
        with tf.name_scope ('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads,tvars))

        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost) 
        

    def sample(self, sess, chars, vocab, num=200, prime=' ', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32)
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        char = prime[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
