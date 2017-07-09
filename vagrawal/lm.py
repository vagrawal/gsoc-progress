import tensorflow as tf
import pywrapfst as fst

def fstCostSingle(poss_states, num_fst_states, inp, LMfst,
        vocab_size, max_states):
    # A very crude try
    poss_states = []
    for st in poss_states:
        for arc in LMfst.arcs(st):
            if arc.ilabel == inp:
                poss_states.append(arc.nextstate)
            if arc.ilabel == '<eps>':
                for arc_ in LMfst.arcs(arc.nextstate):
                    if arc_.ilabel == inp:
                        poss_states.append(arc_.nextstate)
    num_states = len(poss_states)
    return poss_states, num_states,

def fstCosts(states, num_fst_states, inputs, LMfst, vocab_size, max_states):
    next_states = states
    next_num_states = num_fst_states
    scores = np.zeros((num_fst_states.shape[0], vocab_size))
    for i in range(num_fst_states.shape[0]):
        next_states[i], next_num_states[i], scores[i] = fstCostSingle(states[i],
                num_fst_states[i], inputs[i], LMfst, vocab_size, max_states)
    return next_states, num_fst_states, [0.0]

class LMCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, dec_cell, fst_path, max_states, reuse, vocab_size):
        super(LMCellWrapper, self).__init__(_reuse=reuse)
        self.dec_cell = dec_cell
        self.fst = fst.Fst.read(fst_path)
        self._output_size = dec_cell.output_size
        self.max_states = max_states
        # LSTM state, FST states and number of FST states
        self._state_size = (dec_cell.state_size,
                tf.TensorShape((max_states)),
                tf.TensorShape((1)))
        self.vocab_size = vocab_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        cell_state, fst_states, num_fst_states = state
        cell_out, cell_state = self.dec_cell(inputs, cell_state)
        fstCosts = lambda s, n, i: fstCosts(s, n, i, self.fst, self.vocab_size, max_states)
        next_state, next_num_states, lm_scores = tf.py_func(fstCosts,
            [fst_states, num_fst_states, inputs], [tf.int64, tf.int32, tf.float32])
        next_state.set_shape(fst_states.shape)
        next_num_states.set_shape(num_fst_states.shape)
        lm_scores.set_shape(cell_out.shape)
        return cell_out, (cell_state, next_state, next_num_states)

    def zero_state(self, batch_size, dtype):
        return (self.dec_cell.zero_state(batch_size, dtype),
                tf.zeros((batch_size, self.max_states), tf.int64),
                tf.ones((batch_size, 1), tf.int32))
