# Adapted from Currie32's text summarization tutorial
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

def process_target_data(target_data, start_token, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    ending = target_data[:, :-1]
    dec_input = tf.concat([tf.fill([batch_size, 1], start_token), ending], 1)

    return dec_input

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob):
    '''Create the encoding layer'''

    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                    input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(rnn_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                    input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
            rnn_inputs = tf.nn.pool(tf.concat(enc_output,2), window_shape=[2], pooling_type='AVG', padding='SAME')
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)

    return enc_output, enc_state

def training_decoding_layer(target_data, output_length, output_layer, vocab_size,
        rnn_size, enc_output, text_length, dec_cell, batch_size, start_token):
    '''Create the training logits'''

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    target_data = process_target_data(target_data, start_token, batch_size)

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)

    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)

    target_data = tf.nn.embedding_lookup(tf.eye(vocab_size), target_data)

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_data,
                                                        sequence_length=output_length,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                              output_time_major=False,
                                                              impute_finished = True)
    return training_logits

def inference_decoding_layer(vocab_size, start_token, end_token, output_layer,
                             max_output_length, batch_size, beam_width,
                             rnn_size, enc_output, text_length, dec_cell):
    '''Create the inference logits'''
    enc_output = tf.contrib.seq2seq.tile_batch(enc_output, beam_width)
    text_length = tf.contrib.seq2seq.tile_batch(text_length, beam_width)

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)

    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)

    start_tokens = tf.fill([batch_size], start_token, name='start_tokens')

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(dec_cell,
                                                             tf.eye(vocab_size),
                                                             start_tokens,
                                                             end_token,
                                                             initial_state,
                                                             beam_width,
                                                             output_layer)

    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            maximum_iterations=max_output_length)

    return inference_logits

def decoding_layer(target_data, enc_output, enc_state, vocab_size, text_length, output_length,
                   max_output_length, rnn_size, vocab_to_int, keep_prob,
                   batch_size, num_layers, beam_width):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    with tf.variable_scope('decoder'):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size,
                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                 input_keep_prob = keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(target_data,
                                                  output_length,
                                                  output_layer,
                                                  vocab_size,
                                                  rnn_size,
                                                  enc_output,
                                                  text_length,
                                                  dec_cell,
                                                  batch_size,
                                                  vocab_to_int['<GO>'])
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(len(vocab_to_int),
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    output_layer,
                                                    max_output_length,
                                                    batch_size,
                                                    beam_width,
                                                    rnn_size,
                                                    enc_output,
                                                    text_length,
                                                    dec_cell)

    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, keep_prob, input_lengths, output_lengths, max_output_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size,
                  beam_width):
    '''Use the previous functions to create the training and inference logits'''

    enc_output, enc_state = encoding_layer(rnn_size, input_lengths, num_layers, input_data, keep_prob)

    training_logits, inference_logits  = decoding_layer(target_data,
                                                        enc_output,
                                                        enc_state,
                                                        vocab_size,
                                                        input_lengths,
                                                        output_lengths,
                                                        max_output_length,
                                                        rnn_size,
                                                        vocab_to_int,
                                                        keep_prob,
                                                        batch_size,
                                                        num_layers,
                                                        beam_width)

    return training_logits, inference_logits
