import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

def encoding_layer(
        rnn_size,
        input_lengths,
        num_layers,
        rnn_inputs,
        keep_prob):
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                    cell_fw,
                    input_keep_prob = keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(
                    rnn_size,
                    initializer=tf.random_uniform_initializer(
                        -0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                    cell_bw,
                    input_keep_prob = keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    rnn_inputs,
                    input_lengths,
                    dtype=tf.float32)

            if layer != num_layers - 1:
                rnn_inputs = tf.concat(enc_output,2)
                # Keep only every second element in the sequence
                rnn_inputs = tf.strided_slice(
                        rnn_inputs,
                        [0, 0, 0],
                        tf.shape(rnn_inputs),
                        [1, 2, 1],
                        begin_mask=7,
                        end_mask=7)
                input_lengths = (input_lengths + 1) / 2
    # Join outputs since we are using a bidirectional RNN
    enc_output = tf.concat(enc_output,2)

    return enc_output, enc_state, input_lengths

def training_decoding_layer(
        output_data,
        output_lengths,
        output_layer,
        vocab_size,
        rnn_size,
        enc_output,
        input_lengths,
        dec_cell,
        batch_size,
        start_token):
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
            rnn_size,
            enc_output,
            input_lengths,
            normalize=False,
            name='BahdanauAttention')

    output_data = tf.concat(
            [tf.fill([batch_size, 1], start_token), output_data[:, :-1]], 1)

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_cell,
            attn_mech,
            rnn_size)

    initial_state = dec_cell.zero_state(
            dtype=tf.float32,
            batch_size=batch_size)

    output_data = tf.nn.embedding_lookup(
            tf.eye(vocab_size),
            output_data)

    training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=output_data,
            sequence_length=output_lengths,
            time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell,
            training_helper,
            initial_state,
            output_layer)

    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            output_time_major=False,
            impute_finished = True)

    return training_logits

def inference_decoding_layer(
        vocab_size,
        start_token,
        end_token,
        output_layer,
        max_output_length,
        batch_size,
        beam_width,
        rnn_size,
        enc_output,
        input_lengths,
        dec_cell):
    enc_output = tf.contrib.seq2seq.tile_batch(
            enc_output,
            beam_width)

    input_lengths = tf.contrib.seq2seq.tile_batch(
            input_lengths,
            beam_width)

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
            rnn_size,
            enc_output,
            input_lengths,
            normalize=False,
            name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(
            dec_cell,
            attn_mech,
            rnn_size)

    initial_state = dec_cell.zero_state(
            dtype=tf.float32,
            batch_size=batch_size * beam_width)

    start_tokens = tf.fill(
            [batch_size],
            start_token,
            name='start_tokens')

    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            dec_cell,
            tf.eye(vocab_size),
            start_tokens,
            end_token,
            initial_state,
            beam_width,
            output_layer)

    predictions, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder,
            output_time_major=False,
            maximum_iterations=max_output_length)

    return predictions

def seq2seq_model(
        input_data,
        output_data,
        keep_prob,
        input_lengths,
        output_lengths,
        max_output_length,
        vocab_size,
        rnn_size,
        num_layers,
        vocab_to_int,
        batch_size,
        beam_width,
        learning_rate):

    enc_output, enc_state, enc_lengths = encoding_layer(
            rnn_size,
            input_lengths,
            num_layers,
            input_data,
            keep_prob)

    with tf.variable_scope('decoder'):
        lstm = tf.contrib.rnn.LSTMCell(
                rnn_size,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        dec_cell = tf.contrib.rnn.DropoutWrapper(
                lstm,
                input_keep_prob=keep_prob)

    output_layer = Dense(
            vocab_size,
            kernel_initializer = tf.truncated_normal_initializer(
                mean = 0.0,
                stddev=0.1))

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(
                output_data,
                output_lengths,
                output_layer,
                vocab_size,
                rnn_size,
                enc_output,
                enc_lengths,
                dec_cell,
                batch_size,
                vocab_to_int['<GO>'])
    with tf.variable_scope("decode", reuse=True):
        predictions = inference_decoding_layer(
                len(vocab_to_int),
                vocab_to_int['<GO>'],
                vocab_to_int['<EOS>'],
                output_layer,
                max_output_length,
                batch_size,
                beam_width,
                rnn_size,
                enc_output,
                enc_lengths,
                dec_cell)

    # Create tensors for the training logits and predictions
    training_logits = tf.identity(
            training_logits.rnn_output,
            name='logits')
    predictions = tf.identity(
            predictions.predicted_ids[:, :, 0],
            name='predictions')

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(
            output_lengths,
            tf.reduce_max(output_lengths),
            dtype=tf.float32,
            name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            output_data,
            masks)

        tf.summary.scalar('cost', cost)

        step = tf.contrib.framework.get_or_create_global_step()

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(
            tf.clip_by_value(grad, -5., 5.), var)
            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, step)

    return training_logits, predictions, train_op, cost, step

