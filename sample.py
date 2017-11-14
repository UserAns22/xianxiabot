import tensorflow as tf

import os
from six.moves import cPickle

import rnn
SAVE_DIR = './output'

num_chars = 500
prime = ' ' #start with Meng Hao!
sample = 1 #

def sample():
    with open(os.path.join(SAVE_DIR, 'conf.pkl'), 'rb') as f:
        vocab_size, chars, vocab = cPickle.load(f)
    rnn.training = False
    model = rnn.Model(vocab_size, False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = os.path.join(SAVE_DIR, 'model.ckpt-2000')
        saver.restore(sess, ckpt)
        print(model.sample(sess, chars, vocab, num=num_chars,prime=prime, sampling_type=sample).encode('utf-8')) 

if __name__ == "__main__":
    sample()
