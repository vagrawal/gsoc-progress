import os
import tensorflow as tf
import numpy as np
from scipy.fftpack import dct
from python_speech_features.base import mfcc
from seq2seq_model import seq2seq_model
import scipy.io.wavfile
import time
from data import *

def train(keep_prob = 0.8,
          max_input_len = 1000,
          max_output_len = 200,
          rnn_size = 256,
          num_layers = 2,
          batch_size = 32,
          learning_rate = 0.0005,
          num_epochs = 5,
          learning_rate_decay = 0.95,
          min_learning_rate = 0.0005,
          display_step = 1, # Check training loss after every display_step batches
          stop_early = 0,
          stop = 3, # If the update loss does not decrease in 3 consecutive update checks, stop training
          checkpoint = "gs://wsj-data/best_model.ckpt")

# Make a graph and it's session
train_graph = tf.Graph()

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

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate_tensor)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch_i in range(1, num_epochs+1):
        batch_loss = 0
        batch_i = 0
        out_file.seek(0)
        while (True):
            batch_i += 1
            try:
                input_batch, output_batch, input_lengths_batch, output_lengths_batch = get_next_batch()
            except:
                print("Epoch {} completed".format(epoch_i))
                break
                
            start_time = time.time()
            
            _, loss = sess.run(
                [train_op, cost],
                {model_input: input_batch,
                 model_output: output_batch,
                 learning_rate_tensor: learning_rate,
                 output_lengths: output_lengths_batch,
                 input_lengths: input_lengths_batch})

            batch_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time
            
            if batch_i % display_step == 0 and batch_i > 0:
                print('Epoch {:>3}/{} Batch {:>4} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              num_epochs, 
                              batch_i,
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0     
                
                logits = sess.run(
                    inference_logits,
                    {model_input: input_batch,
                     input_lengths: input_lengths_batch})
                real_out = ''.join([vocab[l] for l in output_batch[0, :output_lengths_batch[0] - 1]])
                pred_out = ''.join([vocab[l] for l in logits[0]])
                print("Real output: ", real_out)
                print("Predicted output: ", pred_out)
                print("WER: ", wer(real_out.split(), pred_out.split()))
                    
        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
