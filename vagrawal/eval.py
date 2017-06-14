
# coding: utf-8

# In[ ]:

import os
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from python_speech_features.base import mfcc
from seq2seq_model import seq2seq_model
import scipy.io.wavfile
import time


# In[ ]:

root = 'gs://wsj-data/wsj0/'
keep_prob = 0.95
max_input_len = 1000
max_output_len = 150
rnn_size = 256
num_layers = 3
batch_size = 24
learning_rate = 0.001
num_epochs = 5

learning_rate_decay = 0.99985
min_learning_rate = 0.0005
display_step = 20 # Check training loss after every display_step batches

checkpoint = "gs://wsj-data/checkpoint37"


# In[ ]:

vocab = np.asarray(list(" '+-.ABCDEFGHIJKLMNOPQRSTUVWXYZ_") + ['<GO>', '<EOS>'])
vocab_to_int = {}

for ch in vocab:
    vocab_to_int[ch] = len(vocab_to_int)


# In[ ]:

# A custom class inheriting tf.gfile.Open for providing seek with whence
class FileOpen(tf.gfile.Open):
    def seek(self, position, whence = 0):
        if (whence == 0):
            tf.gfile.Open.seek(self, position)
        elif (whence == 1):
            tf.gfile.Open.seek(self, self.tell() + position)
        else:
            raise FileError


# In[ ]:

# https://github.com/zszyellow/WER-in-python/blob/master/wer.py
def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split())
    """
    #build the matrix
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0: d[0][j] = j
            elif j == 0: d[i][0] = i
    for i in range(1,len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return float(d[len(r)][len(h)]) / max(len(r), len(h)) * 100


# In[ ]:

out_file = tf.gfile.Open(root + 'transcripts/wsj0/wsj0.trans')
numcep = 13

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
    return [[vocab_to_int['<GO>']] + sentence + [vocab_to_int['<EOS>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

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
        output_batch_length[i] = len(out) + 2
    output_batch = np.asarray(pad_sentence_batch(output_batch))
    return (input_batch, output_batch, input_batch_length, output_batch_length)


# In[ ]:

# Make a graph and it's session
train_graph = tf.Graph()
#train_session = tf.InteractiveSession(graph=train_graph)

# Set the graph to default to ensure that it is ready for training
with train_graph.as_default():
    model_input = tf.placeholder(tf.float32, [batch_size, max_input_len, numcep], name='model_input')
    model_output = tf.placeholder(tf.int32, [batch_size, None], name='model_output')
    input_lengths = tf.placeholder(tf.int32, [batch_size], name='input_lengths')
    output_lengths = tf.placeholder(tf.int32, [batch_size], name='output_lengths')
    learning_rate_tensor = tf.placeholder(tf.float32, name='learning_rate')

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(input_data=model_input,
                                                      target_data=model_output,
                                                      keep_prob=keep_prob,
                                                      input_lengths=input_lengths,
                                                      output_lengths=output_lengths,
                                                      max_output_length=max_output_len,
                                                      vocab_size=len(vocab_to_int),
                                                      rnn_size=rnn_size,
                                                      num_layers=num_layers,
                                                      vocab_to_int=vocab_to_int,
                                                      batch_size=batch_size)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(output_lengths, tf.reduce_max(output_lengths), dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            model_output,
            masks)

        tf.summary.scalar('cost', cost)

        step = tf.contrib.framework.get_or_create_global_step()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate_tensor)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, step)


# In[ ]:

with train_graph.as_default():
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint,
              save_checkpoint_secs=600,
              save_summaries_steps=20) as sess:

        # If we want to continue training a previous session
        #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        #loader.restore(sess, checkpoint)
        writer = tf.summary.FileWriter(checkpoint)

        batch_loss = 0.0
        for epoch_i in range(1, num_epochs+1):
            out_file.seek(0)
            while (True):
                try:
                    input_batch, output_batch, input_lengths_batch, output_lengths_batch = get_next_batch()
                except:
                    #tf.logging.info("Error: {}".format(sys.exc_info()[0]))
                    tf.logging.info("Epoch {} (likely) completed".format(epoch_i))
                    break

                logits = sess.run(
                    inference_logits,
                    {model_input: input_batch,
                     input_lengths: input_lengths_batch})
                tot_wer = 0.0
                tot_cer = 0.0
                for i in range(batch_size):
                    real_out = ''.join([vocab[l] for l in output_batch[i, :output_lengths_batch[i] - 1]])
                    pred_out = ''.join([vocab[l] for l in logits[i]])
                    #pred_out = pred_out.split('<')[0]
                    tot_wer += wer(real_out.split(), pred_out.split())
                    tot_cer += wer(list(real_out), list(pred_out))
                tf.logging.info('Sample real output: {}'.format(real_out))
                tf.logging.info('Sample predicted output: {}'.format(pred_out))
                tf.logging.info('WER: {}, CER: {}'.format(tot_wer / batch_size, tot_cer / batch_size))

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate


# In[ ]:

get_next_batch()


# In[ ]:



