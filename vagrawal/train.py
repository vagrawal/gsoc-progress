import os
import tensorflow as tf
import numpy as np
from seq2seq_model import seq2seq_model
from data import read_data_queue, get_speaker_stats
from vocab import vocab
from vocab import vocab
import time
import argparse
from fst import fstCosts

def run_eval(graph, job_dir, checkpoint, queue, predictions, data_dir, nfilt,
        sess, coord, outputs, output_lengths,
        batch_i, cost, keep_prob_tensor, mean_speaker, var_speaker,
        best_n_inference, pred_scores, LMfst):

    tf.logging.info("Evaluation started")
    with graph.as_default():
        writer = tf.summary.FileWriter(job_dir)
        tf.Session.reset(None, ['queue'])
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, checkpoint)
            read_data_queue('si_et_05', queue, data_dir, nfilt,
                    sess, mean_speaker, var_speaker, LMfst)
            tot_wer = 0.0
            tot_cer = 0.0
            batch_loss = 0.0
            tot_ev = 0
            tot_bat = 0
            coord = tf.train.Coordinator(
                    clean_stop_exception_types=(
                        tf.errors.CancelledError,
                        tf.errors.OutOfRangeError))

            with coord.stop_on_exception():
                while not coord.should_stop():
                    pred, sc, out, out_len, loss = sess.run(
                            [predictions, pred_scores, outputs, output_lengths, cost],
                            feed_dict={keep_prob_tensor: 1.0})
                    tot_ev += pred.shape[0]
                    tot_bat += 1
                    batch_loss += loss
                    for i in range(pred.shape[0]):
                        best_wer = 100.0
                        best_cer = 100.0
                        for j in range(best_n_inference):
                            real_out = ''.join([vocab[l] for l in out[i, :out_len[i] - 1]])
                            pred_out = ''.join([vocab[l] for l in pred[i, :, j]])
                            pred_out = pred_out.split('<')[0]
                            cur_wer = wer(real_out.split(), pred_out.split())
                            tf.logging.info("{} : {}".format(pred_out, sc[i, j]))
                            best_wer = min(best_wer, cur_wer)
                            best_cer = min(best_cer, wer(list(real_out), list(pred_out)))
                        tot_wer += best_wer
                        tot_cer += best_cer

            tf.logging.info('WER: {}, CER: {}'.format(tot_wer / tot_ev, tot_cer / tot_ev))
            summary = tf.Summary(value=
                [tf.Summary.Value(tag="WER_valid", simple_value=tot_wer / tot_ev),
                 tf.Summary.Value(tag="CER_valid", simple_value=tot_cer / tot_ev),
                 tf.Summary.Value(tag="loss_valid", simple_value=batch_loss / tot_bat)
                ])
            writer.add_summary(summary, global_step=batch_i)
            writer.flush()
            coord.request_stop()
    tf.logging.info("Evaluation finished")

def train(
        nfilt,
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
        job_dir,
        checkpoint_path,
        best_n_inference,
        eval_only,
        fst_path):

    checkpoint = job_dir + 'checkpoints/'

    if eval_only:
        sets = ['si_et_05']
    else:
        sets = ['si_et_05', 'si_tr_s']

    LMfst = fst.Fst.read_from_string(FileOpen(fst_path).read())

    graph = tf.Graph()
    with graph.as_default():
        learning_rate_tensor = tf.placeholder(
                tf.float32,
                name='learning_rate')
        keep_prob_tensor = tf.placeholder(
                tf.float32,
                name='keep_prob')
        # https://stackoverflow.com/questions/39204335/can-a-tensorflow-queue-be-reopened-after-it-is-closed
        with tf.container('queue'):
            queue = tf.PaddingFIFOQueue(
                capacity=64,
                dtypes=['float32', 'int32', 'int32', 'int32'],
                shapes=[[None, nfilt * 3 + 1], [], [None], []],
                name='feed_queue')
            inputs, input_lengths, outputs, output_lengths = queue.dequeue_many(batch_size)

        training_logits, predictions, train_op, cost, step, pred_scores = seq2seq_model(
                inputs,
                outputs,
                keep_prob_tensor,
                input_lengths,
                output_lengths,
                max_output_len,
                rnn_size,
                num_layers,
                tf.shape(input_lengths)[0],
                beam_width,
                learning_rate_tensor,
                LMfst)

        writer = tf.summary.FileWriter(job_dir)
        saver = tf.train.Saver()
        batch_loss = 0.0

        mean_speaker, var_speaker = get_speaker_stats(data_dir, nfilt, sets)

        for epoch_i in range(1, num_epochs + 1):
            tf.Session.reset(None, ['queue'])
            with tf.Session() as sess:
                coord = tf.train.Coordinator(
                        clean_stop_exception_types=(
                            tf.errors.CancelledError,
                            tf.errors.OutOfRangeError))
                if (checkpoint_path is None):
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                else:
                    saver.restore(sess, checkpoint_path)
                    batch_i = sess.run(step)
                    run_eval(graph, job_dir, checkpoint_path, queue, predictions, data_dir,
                            nfilt, sess, coord, outputs,
                            output_lengths, batch_i, cost, keep_prob_tensor,
                            mean_speaker, var_speaker, best_n_inference,
                            pred_scores, LMfst)
                    if (eval_only):
                        coord.request_stop()
                        return

                read_data_queue('si_tr_s', queue, data_dir, nfilt,
                        sess, None, None, LMfst)
                        sess, mean_speaker, var_speaker)

                with coord.stop_on_exception():
                    while not coord.should_stop():
                        start_time = time.time()

                        loss, _, batch_i = sess.run(
                                [cost, train_op, step],
                                feed_dict={learning_rate_tensor: learning_rate,
                                    keep_prob_tensor: keep_prob})
                        print("Loss: ", loss)

                        batch_loss += loss
                        end_time = time.time()
                        batch_time = end_time - start_time

                        if batch_i % display_step == 0 and batch_i > 0:
                            tf.logging.info('Epoch {:>3}/{} Batch {:>4} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                                  .format(epoch_i,
                                          num_epochs,
                                          batch_i,
                                          batch_loss / display_step,
                                          batch_time))
                            tot_wer = 0.0
                            tot_cer = 0.0

                            pred, out, out_len = sess.run(
                                    [predictions, outputs, output_lengths],
                                    feed_dict={keep_prob_tensor: 1.0})
                            for i in range(pred.shape[0]):
                                real_out = ''.join([vocab[l] for l in out[i, :out_len[i] - 1]])
                                pred_out = ''.join([vocab[l] for l in pred[i, :, 0]])
                                pred_out = pred_out.split('<')[0]
                                tot_wer += wer(real_out.split(), pred_out.split())
                                tot_cer += wer(list(real_out), list(pred_out))
                            tf.logging.info('Sample real output: {}'.format(real_out))
                            tf.logging.info('Sample predicted output: {}'.format(pred_out))
                            tf.logging.info('WER: {}, CER: {}'.format(tot_wer / pred.shape[0], tot_cer / pred.shape[0]))
                            summary = tf.Summary(value=
                                [tf.Summary.Value(tag="WER", simple_value=tot_wer / pred.shape[0]),
                                 tf.Summary.Value(tag="CER", simple_value=tot_cer / pred.shape[0]),
                                 tf.Summary.Value(tag="loss", simple_value=batch_loss / display_step)
                                ])
                            writer.add_summary(summary, global_step=batch_i)
                            writer.flush()
                            batch_loss = 0.0

                        # Reduce learning rate, but not below its minimum value
                        learning_rate *= learning_rate_decay
                        if learning_rate < min_learning_rate:
                            learning_rate = min_learning_rate

                tf.logging.info("Epoch completed, saving")
                checkpoint_path = saver.save(
                        sess, checkpoint, step)

                coord.request_stop()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--nfilt", default=40, type=int)
    parser.add_argument("--keep-prob", default=0.8, type=float)
    parser.add_argument("--max-output-len",  default=200, type=int)
    parser.add_argument("--rnn-size", default=256, type=int)
    parser.add_argument("--num-layers",  default=3, type=int)
    parser.add_argument("--batch-size",  default=24, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--num-epochs", default=16, type=int)
    parser.add_argument("--beam-width", default=8, type=int)
    parser.add_argument("--learning-rate-decay", default=0.9998, type=float)
    parser.add_argument("--min-learning-rate", default=0.0002, type=float)
    parser.add_argument("--display-step", default=20, type=int) # Check training loss after every display_step batches
    parser.add_argument("--data-dir", default='gs://wsj-data/wsj0/')
    parser.add_argument("--job-dir", default='./job/')
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--best-n-inference", default=1, type=int)
    parser.add_argument("--eval-only", default=False, type=bool)
    parser.add_argument("--fst-path")
    args = parser.parse_args()
    train(**args.__dict__)
