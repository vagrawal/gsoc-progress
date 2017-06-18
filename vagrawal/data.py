import numpy as np
from python_speech_features.base import mfcc
import scipy.io.wavfile
import tensorflow as tf

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
    features = mfcc(signal, sample_rate, numcep=numcep)
    return (features, features.shape[0])

def read_data(root, numcep, batch_size, vocab_to_int, num_epochs):
    trans = tf.gfile.Open(root + 'transcripts/wsj0/wsj0.trans').readlines()
    train_texts = []
    train_files = []
    valid_texts = []
    valid_files = []
    for line in trans:
        text, file = line.split('(')
        text = text[:-1]
        text = [vocab_to_int[c] for c in list(text)] + vocab_to_int['<EOS>']
        file = file[:-2]
        # Training set
        if ('si_tr_s' == file.split('/')[2] and 'wv1' == file.split('.')[1]):
            train_texts.append(text)
            train_files.append(root + 'wav/' + file)
        # Validation set
        if ('sd_et_20' == file.split('/')[2] and 'wv1' == file.split('.')[1]):
            valid_texts.append(text)
            valid_files.append(root + 'wav/' + file)
    train_texts, train_files = tf.train.slice_input_producer([train_texts,
        train_files], num_epochs=num_epochs)
    train_featues = tf.py_func(lambda x: get_features(x, numcep), [train_files],
            tf.float32, stateful=False)
    train_texts, train_files = tf.train.batch([train_texts, train_files],
            batch_size, shapes=[[None], [None, None]], dynamic_pad=True)

    valid_texts, valid_files = tf.train.slice_input_producer([valid_texts,
        valid_files], num_epochs=num_epochs)
    valid_featues = tf.py_func(lambda x: get_features(x, numcep), [valid_files],
            tf.float32, stateful=False)
    valid_texts, valid_files = tf.train.batch([valid_texts, valid_files],
            batch_size, shapes=[[None], [None, None]], dynamic_pad=True)
    return train_texts, train_featues, valid_texts, valid_files
