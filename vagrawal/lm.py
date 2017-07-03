
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

def next_symbol_probs(
        rnn_size,
        lengths,
        num_layers,
        rnn_inputs,
        keep_prob):

    cell = tf.contrib.rnn.LSTMCell(
            rnn_size,
            initializer=tf.random_uniform_initializer(
                -0.1, 0.1, seed=2))
    cell = tf.contrib.rnn.DropoutWrapper(
            cell_fw,
            input_keep_prob = keep_prob)

    cell = tf.contrib.rnn.MultiRNNCell(
            [cell for _ in range(num_layers)], state_is_tuple=True)


def language_model(
        data,
        keep_prob,
        lengths,
        vocab_size,
        rnn_size,
        num_layers,
        vocab_to_int,
        batch_size,
        beam_width,
        learning_rate):

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

