import tensorflow as tf
import pywrapfst as fst
from itertools import groupby
from data import vocab_size, FileOpen
import numpy as np

from data import vocab, vocab_to_int

def dfsProbs(state, prob, LMfst, probsArr):
    for arc in LMfst.arcs(state):
        curProb = prob + float(arc.weight.to_string())
        if arc.ilabel == vocab_to_int['<backoff>']:
            dfsProbs(arc.nextstate, curProb, LMfst, probsArr)
            continue
        probsArr[arc.ilabel].append(curProb)

def dfsFind(state, prob, LMfst, inp, out):
    for arc in LMfst.arcs(state):
        curProb = prob + float(arc.weight.to_string())
        if arc.ilabel == vocab_to_int['<backoff>']:
            dfsFind(arc.nextstate, curProb, LMfst, inp, out)
            continue
        if arc.ilabel == inp:
            out.append((arc.nextstate, curProb))

def combine(probs):
    if (len(probs) == 0):
        return 50
    return -np.log(np.sum(np.exp(-np.array(probs))))

def fstCostSingle(poss_states, probs, num_fst_states, inp, LMfst, max_states):
    # A very crude try
    next_tup = []
    for i in range(num_fst_states):
        dfsFind(poss_states[i], probs[i], LMfst, inp, next_tup)
    next_tup = groupby(sorted(next_tup), lambda t: t[0])
    num_states = 0
    next_states = []
    next_probs = []
    scores = []
    probsArr = [[] for _ in range(vocab_size)]
    for st, we in next_tup:
        num_states += 1
        next_states.append(st)
        prob = combine([w[1] for w in we])
        next_probs.append(prob)
        dfsProbs(st, prob, LMfst, probsArr)
    for i in range(vocab_size):
        probsArr[i] = combine(probsArr[i])
    while(len(next_states) != max_states):
        next_states.append(0)
        next_probs.append(0)
    return np.asarray(next_states), np.asarray(next_probs), num_states, np.asarray(probsArr)

def fstCosts(states, state_probs, num_fst_states, inputs, LMfst, max_states):
    next_states = states
    next_state_probs = state_probs
    next_num_states = num_fst_states
    scores = np.zeros((num_fst_states.shape[0], vocab_size))
    for i in range(num_fst_states.shape[0]):
        next_states[i], next_state_probs[i], next_num_states[i], scores[i] = fstCostSingle(states[i],
                state_probs[i], num_fst_states[i], inputs[i], LMfst, max_states)
    return next_states, next_state_probs, num_fst_states, scores

class LMCellWrapper(tf.contrib.rnn.RNNCell):
    def __init__(self, dec_cell, fst_path, max_states, reuse):
        super(LMCellWrapper, self).__init__(_reuse=reuse)
        self.dec_cell = dec_cell
        self.fst = fst.Fst.read_from_string(FileOpen(fst_path).read())
        self._output_size = dec_cell.output_size
        self.max_states = max_states
        # LSTM state, FST states and number of FST states
        self._state_size = (dec_cell.state_size,
                tf.TensorShape((max_states)),
                tf.TensorShape((max_states)),
                tf.TensorShape((1)))

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state):
        cell_state, fst_states, state_probs, num_fst_states = state
        cell_out, cell_state = self.dec_cell(inputs, cell_state)
        fstCosts = lambda s, p, n, i: fstCosts(s, p, n, i, self.fst, max_states)
        next_state, next_state_probs, next_num_states, lm_scores = tf.py_func(fstCosts,
            [fst_states, state_probs, num_fst_states, tf.argmax(inputs)], [tf.int64,
                tf.float32, tf.int32, tf.float32])
        next_state.set_shape(fst_states.shape)
        next_num_states.set_shape(num_fst_states.shape)
        next_state_probs.set_shape(state_probs.shape)
        lm_scores.set_shape(cell_out.shape)
        next_state_probs = tf.nn.log_softmax(next_state_probs)
        fin_score = tf.nn.log_softmax(cell_out) + tf.nn.log_softmax(lm_scores)
        return fin_score, (cell_state, next_state, next_state_probs, next_num_states)

    def zero_state(self, batch_size, dtype):
        return (self.dec_cell.zero_state(batch_size, dtype),
                tf.zeros((batch_size, self.max_states), tf.int64),
                tf.zeros((batch_size, self.max_states), tf.float32),
                tf.ones((batch_size, 1), tf.int32))
