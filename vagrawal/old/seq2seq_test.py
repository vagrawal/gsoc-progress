import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense

encoder_sequence_length = np.array([3, 2, 3, 1, 1])
decoder_sequence_length = np.array([2, 0, 1, 2, 3])
batch_size = 5
decoder_max_time = 4
input_depth = 7
cell_depth = 9
attention_depth = 6
vocab_size = 20
end_token = vocab_size - 1
start_token = 0
embedding_dim = 50
max_out = max(decoder_sequence_length)
output_layer = Dense(vocab_size, use_bias=True, activation=None)
beam_width = 3

with tf.Session() as sess:
  batch_size_tensor = tf.constant(batch_size)
  embedding = tf.constant(np.random.randn(vocab_size, embedding_dim).astype(np.float32))
  cell = tf.contrib.rnn.LSTMCell(cell_depth)
  initial_state = cell.zero_state(batch_size, tf.float32)
  if True:
    inputs = tf.placeholder_with_default(
        np.random.randn(batch_size, decoder_max_time,
                        input_depth).astype(np.float32),
        shape=(None, None, input_depth))
    tiled_inputs = tf.contrib.seq2seq.tile_batch(
        inputs, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        encoder_sequence_length, multiplier=beam_width)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    initial_state = tf.contrib.seq2seq.tile_batch(
        initial_state, multiplier=beam_width)
    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=cell,
        attention_mechanism=attention_mechanism,
        attention_layer_size=attention_depth,
        alignment_history=False)
  cell_state = cell.zero_state(
      dtype=tf.float32, batch_size=batch_size_tensor * beam_width)
  if True:
    cell_state = cell_state.clone(
        cell_state=initial_state)
  bsd = tf.contrib.seq2seq.BeamSearchDecoder(
      cell=cell,
      embedding=embedding,
      start_tokens=tf.fill([batch_size_tensor], start_token),
      end_token=end_token,
      initial_state=cell_state,
      beam_width=beam_width,
      output_layer=output_layer,
      length_penalty_weight=0.0)

  final_outputs, final_state, final_sequence_lengths = (
      tf.contrib.seq2seq.dynamic_decode(
          bsd, output_time_major=False, maximum_iterations=100))
