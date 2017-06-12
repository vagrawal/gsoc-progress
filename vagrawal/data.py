import os
import numpy as np
from scipy.fftpack import dct
from python_speech_features.base import mfcc
import scipy.io.wavfile

root = 'gs://wsj-data/wsj0/'

# A custom class inheriting tf.gfile.Open for providing seek with whence
class FileOpen(tf.gfile.Open):
    def seek(self, position, whence = 0):
        if (whence == 0):
            tf.gfile.Open.seek(self, position)
        elif (whence == 1):
            tf.gfile.Open.seek(self, self.tell() + position)
        else:
            raise FileError

out_file = tf.gfile.Open(root + 'transcripts/wsj0/wsj0.trans')

def get_next_input():
    trans = out_file.readline()
    cont, file = trans.split('(')
    file = file[:-2]
    sample_rate, signal = scipy.io.wavfile.read(FileOpen(root + 'wav/' + file.rstrip('\n'), 'rb'))
    Y = [vocab_to_int[c] for c in list(cont)]
    X = mfcc(signal, sample_rate, numcep=numcep)
    return X, Y

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <EOS> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch]) + 1
    return [sentence + [vocab_to_int['<EOS>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_next_batch():
    input_batch = np.zeros((batch_size, max_input_len, numcep), 'float32')
    output_batch = [] # Variable shape of maximum length string 
    input_batch_length = np.zeros((batch_size), 'int')
    output_batch_length = np.zeros((batch_size), 'int')
    for i in range(batch_size):
        inp, out = get_next_input()
        inp = inp[:max_input_len]
        input_batch[i, :inp.shape[0]] = inp
        output_batch.append(out)
        input_batch_length[i] = inp.shape[0]
        output_batch_length[i] = len(out) + 1
    output_batch = np.asarray(pad_sentence_batch(output_batch))
    return (input_batch, output_batch, input_batch_length, output_batch_length)