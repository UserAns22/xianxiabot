import tensorflow as tf

import time
import os
from six.moves import cPickle

from utils import TextLoader
import rnn


NUM_EPOCHS = 2
DATA_DIR = './data'
SAVE_DIR = './output'
init_from = SAVE_DIR


def train():
    loader = TextLoader(DATA_DIR, rnn.BATCH_SIZE, rnn.SEQ_LENGTH)
    vocab_size = loader.vocab_size
    
    if init_from is not None:
        ckpt = tf.train.get_checkpoint_state(init_from)
        
    with open(os.path.join(SAVE_DIR + 'conf.pkl'), 'wb') as f:
        cPickle.dump((loader.vocab_size, loader.chars, loader.vocab), f)

    model = rnn.Model(vocab_size,True)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

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
                
                train_loss, state, _= sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print(('{0}/{1} (epoch {2}),' +
                    ' train_loss = {3:.2f},' + 
                    ' time/batch = {4:.2f},' +
                    ' time_left = {5:.2f}').format(
                    curr_batch, NUM_EPOCHS * loader.num_batches, i, 
                    train_loss, end - start,
                    ((end - start) / 3600 * (NUM_EPOCHS * loader.num_batches - curr_batch))))
                if curr_batch % 1000 == 0 or ( i == (NUM_EPOCHS - 1) and (b == loader.num_batches - 1)):
                    ckpath = os.path.join(SAVE_DIR, 'model.ckpt')
                    saver.save(sess, ckpath, global_step = curr_batch)
                    print('model saved to {}'.format(ckpath))

if __name__ == "__main__":
    train()


