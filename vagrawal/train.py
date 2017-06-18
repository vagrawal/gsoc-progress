import os
import tensorflow as tf
import numpy as np
from seq2seq_model import seq2seq_model
from data import read_data_queue
import time
import argparse

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



def run_eval(graph, checkpoint, queue, predictions, data_dir, numcep,
        vocab_to_int, sess, coord, outputs, output_lengths, vocab,
        step):

    tf.logging.info("Evaluation started")
    with graph.as_default():
        writer = tf.summary.FileWriter(checkpoint)
        with tf.Session() as sess:
            tf.train.Saver().saver.restore(sess, checkpoint)
            read_data_queue('sd_et_20', queue, data_dir, numcep, vocab_to_int, sess)
            tot_wer = 0.0
            tot_cer = 0.0
            tot_ev = 0

            with coord.stop_on_exception():
                while not coord.should_stop():
                    pred, out, out_len = sess.run([predictions, outputs, output_lengths])
                    total_ev += pred.shape[0]
                    for i in range(pred.shape[0]):
                        real_out = ''.join([vocab[l] for l in out[i, :out_len[i] - 1]])
                        pred_out = ''.join([vocab[l] for l in pred[i, :]])
                        pred_out = pred_out.split('<')[0]
                        tot_wer += wer(real_out.split(), pred_out.split())
                        tot_cer += wer(list(real_out), list(pred_out))
                        tot_ev += pred.shape[0]
            summary = tf.Summary(value=
                [tf.Summary.Value(tag="WER_valid", simple_value=tot_wer / tot_ev),
                 tf.Summary.Value(tag="CER_valid", simple_value=tot_cer / tot_ev),
                 tf.Summary.Value(tag="loss_valid", simple_value=batch_loss / tot_ev)
                ])
            writer.add_summary(summary, global_step=batch_i)
            writer.flush()
            coord.request_stop()
    tf.logging.info("Evaluation finished")

def train(
        numcep,
        keep_prob,
        max_output_len,
        rnn_size,
        num_layers,
        batch_size,
        learning_rate,
        num_epochs,
        beam_width,
        learning_rate_decay,
        min_learning_rate,
        display_step,
        data_dir,
        job_dir):

    vocab = np.asarray(
            list(" '+-.ABCDEFGHIJKLMNOPQRSTUVWXYZ_") + ['<GO>', '<EOS>'])
    vocab_to_int = {}

    for ch in vocab:
        vocab_to_int[ch] = len(vocab_to_int)

    checkpoint = job_dir + 'checkpoints/'

    graph = tf.Graph()
    with graph.as_default():
        learning_rate_tensor = tf.placeholder(
                tf.float32,
                name='learning_rate')
        # https://stackoverflow.com/questions/39204335/can-a-tensorflow-queue-be-reopened-after-it-is-closed
        with tf.container('queue'):
            queue = tf.PaddingFIFOQueue(
                capacity=64,
                dtypes=['float32', 'int32', 'int32', 'int32'],
                shapes=[[None, numcep], [], [None], []],
                name='feed_queue')
            inputs, input_lengths, outputs, output_lengths = queue.dequeue_up_to(batch_size)

        training_logits, predictions, train_op, cost, step = seq2seq_model(
                inputs,
                outputs,
                keep_prob,
                input_lengths,
                output_lengths,
                max_output_len,
                len(vocab),
                rnn_size,
                num_layers,
                vocab_to_int,
                tf.shape(input_lengths)[0],
                beam_width,
                learning_rate_tensor)

        writer = tf.summary.FileWriter(checkpoint)
        saver = tf.train.Saver()

        for epoch_i in range(1, num_epochs):
            with tf.Session() as sess:
                if (epoch_i == 1):
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                else:
                    saver.restore(sess, checkpoint_path)
                coord = tf.train.Coordinator(
                        clean_stop_exception_types=(
                            tf.errors.CancelledError,
                            tf.errors.OutOfRangeError))

                batch_loss = 0.0
                read_data_queue('si_tr_s', queue, data_dir, numcep,
                        vocab_to_int, sess)

                with coord.stop_on_exception():
                    while not coord.should_stop():
                        start_time = time.time()

                        batch_i, _, loss = sess.run(
                                [step, train_op, cost],
                                feed_dict={learning_rate_tensor: learning_rate})
                        print(loss)

                        batch_loss += loss
                        end_time = time.time()
                        batch_time = end_time - start_time

                        if batch_i % display_step == 0 and batch_i > 0:
                            tf.logging.info('Epoch {:>3}/{} Batch {:>4} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                                  .format(epoch_i,
                                          num_epochs,
                                          batch_i,
                                          batch_loss / display_step,
                                          batch_time * display_step))
                            batch_loss = 0.0
                            tot_wer = 0.0
                            tot_cer = 0.0

                            pred, out, out_len = sess.run([predictions, outputs, output_lengths])
                            for i in range(pred.shape[0]):
                                real_out = ''.join([vocab[l] for l in out[i, :out_len[i] - 1]])
                                pred_out = ''.join([vocab[l] for l in pred[i, :]])
                                pred_out = pred_out.split('<')[0]
                                tot_wer += wer(real_out.split(), pred_out.split())
                                tot_cer += wer(list(real_out), list(pred_out))
                            tf.logging.info('Sample real output: {}'.format(real_out))
                            tf.logging.info('Sample predicted output: {}'.format(pred_out))
                            tf.logging.info('WER: {}, CER: {}'.format(tot_wer / pred.shape[0], tot_cer / pred.shape[0]))
                            summary = tf.Summary(value=
                                [tf.Summary.Value(tag="WER", simple_value=tot_wer / pred.shape[0]),
                                 tf.Summary.Value(tag="CER", simple_value=tot_cer / pred.shape[0]),
                                 tf.Summary.Value(tag="loss", simple_value=batch_loss / pred.shape[0])
                                ])
                            writer.add_summary(summary, global_step=batch_i)
                            writer.flush()

                        # Reduce learning rate, but not below its minimum value
                        learning_rate *= learning_rate_decay
                        if learning_rate < min_learning_rate:
                            learning_rate = min_learning_rate

                tf.logging.info("Epoch completed, saving")
                checkpoint_path = saver.save(
                        sess, checkpoint, step, "Epoch_{}".format(epoch_i))
                run_eval(graph, checkpoint_path, queue, predictions, data_dir,
                        numcep, vocab_to_int, sess, coord, outputs,
                        output_lengths, vocab, step)
                coord.request_stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numcep", default=13)
    parser.add_argument("--keep-prob", default=0.8)
    parser.add_argument("--max-output-len",  default=200)
    parser.add_argument("--rnn-size", default=256)
    parser.add_argument("--num-layers",  default=3)
    parser.add_argument("--batch-size",  default=24)
    parser.add_argument("--learning-rate", default=0.001)
    parser.add_argument("--num-epochs", default=16)
    parser.add_argument("--beam-width", default=8)
    parser.add_argument("--learning-rate-decay", default=0.9998)
    parser.add_argument("--min-learning-rate", default=0.0002)
    parser.add_argument("--display-step", default=10) # Check training loss after every display_step batches
    parser.add_argument("--data-dir", default='gs://wsj-data/wsj0/')
    parser.add_argument("--job-dir", default='./job/')
    args = parser.parse_args()
    train(**args.__dict__)
