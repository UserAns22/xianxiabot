import tensorflow as tf

import time
import os
from six.moves import cPickle

from utils import TextLoader
import rnn

NUM_EPOCHS = 32
DATA_DIR = './data'
SAVE_DIR = './output'



def train():
    loader = TextLoader(DATA_DIR, rnn.BATCH_SIZE, rnn.SEQ_LENGTH)
    vocab_size = loader.vocab_size
    model = rnn.Model(vocab_size)
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        for i in range(NUM_EPOCHS):
            sess.run(tf.assign(model.lr, rnn.learning_rate * (rnn.decay_rate ** i)))
            loader.reset_batch_pointer()

            state = sess.run(model.initial_state)
            for b in range(loader.num_batches):
                curr_batch = i * loader.num_batches + b
                start = time.time()
                x, y = loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for j, s in enumerate(model.initial_state):
                    feed[s] = state[j]
                
                train_loss, state = sess.run([model.cost, model.final_state], feed)
                end = time.time()
                print('{}/{} (epoch {}), train_loss = {}, time/batch = {}'.format(
                    curr_batch, NUM_EPOCHS * loader.num_batches, i, train_loss, end - start))
                if curr_batch % 1000 == 0 or ( i == (NUM_EPOCHS - 1) and (b == loader.num_batches - 1)):
                    ckpath = os.path.join(SAVE_DIR, 'model.ckpt')
                    saver.save(sess, ckpath, global_step = curr_batch)
                    print('model saved to {}'.format(ckpath))

if __name__ == "__main__":
    train()


