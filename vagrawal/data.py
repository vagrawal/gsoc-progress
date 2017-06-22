import numpy as np
from python_speech_features.base import mfcc
import scipy.io.wavfile
import tensorflow as tf
import threading
import random

# A custom class inheriting tf.gfile.Open for providing seek with whence
class FileOpen(tf.gfile.Open):
    def seek(self, position, whence = 0):
        if (whence == 0):
            tf.gfile.Open.seek(self, position)
        elif (whence == 1):
            tf.gfile.Open.seek(self, self.tell() + position)
        else:
            raise FileError

def get_features(file, numcep):
    sample_rate, signal = scipy.io.wavfile.read(FileOpen(file))
    features = mfcc(signal, sample_rate, numcep=numcep, nfilt=2*numcep)
    return features

def read_data(root, numcep, vocab_to_int, sess):
    train_queue = tf.PaddingFIFOQueue(
        capacity=64,
        dtypes=['float32', 'int32', 'int32', 'int32'],
        shapes=[[None, numcep], [], [None], []])
    read_data_queue('si_tr_s', train_queue, root, numcep, vocab_to_int, sess)

    valid_queue = tf.PaddingFIFOQueue(
        capacity=64,
        dtypes=['float32', 'int32', 'int32', 'int32'],
        shapes=[[None, numcep], [], [None], []])
    read_data_queue('sd_et_20', valid_queue, root, numcep, vocab_to_int, sess)

    return train_queue, valid_queue

def read_data_queue(set_id, queue, root, numcep, vocab_to_int, sess):
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, numcep])
    input_length = tf.placeholder(dtype=tf.int32, shape=[])
    output_data = tf.placeholder(dtype=tf.int32, shape=[None])
    output_length =  tf.placeholder(dtype=tf.int32, shape=[])
    enqueue_op = queue.enqueue(
            [input_data, input_length, output_data, output_length])
    close_op = queue.close()

    thread = threading.Thread(
        target=read_data_thread,
        args=(
            set_id,
            root,
            numcep,
            vocab_to_int,
            queue,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op,
            close_op))
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()

def read_data_thread(
        set_id,
        root,
        numcep,
        vocab_to_int,
        queue,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op,
        close_op):
    trans = FileOpen(root + 'transcripts/wsj0/wsj0.trans').readlines()
    random.shuffle(trans)
    for line in trans:
        text, file = line.split('(')
        text = text[:-1]
        # Remove sounds
        text = "".join(text.split("++")[::2])
        text = [vocab_to_int[c] for c in list(text)] + [vocab_to_int['<EOS>']]
        file = file[:-2]
        # Training set
        if (set_id == file.split('/')[2] and 'wv1' == file.split('.')[1]):
            feat = get_features(root + 'wav/' + file, numcep)
            feat = feat - np.mean(feat, 0)
            feat = feat / np.sqrt(np.var(feat, 0))
            sess.run(enqueue_op, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length:len(text)})
    sess.run(close_op)

