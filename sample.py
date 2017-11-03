import tensorflow as tf

import os
from six.moves import cPickle

import rnn
from train import SAVE_DIR

num_chars = 500
prime = 'M' #start with Meng Hao!
sample = 1 #

def sample():
    with open(os.path.join(SAVE_DIR, 'conf.pkl'), 'rb') as f:
        size, chars, vocab = cPickle.load(f)
    rnn.training = False
    model = rnn.Model(size)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(SAVE_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(model.sample(sess, chars, vocab, num_chars,sample).encode('utf-8')) 

if __name__ == "__main__":
    sample()
