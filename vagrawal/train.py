import os
import tensorflow as tf
import numpy as np
from seq2seq_model import get_model
import time

def run_eval(checkpoint, graph, step):
    with graph.as_default():
        writer = tf.summary.FileWriter(self.job_dir)
        with tf.Session(graph=graph) as session:
            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())
            tf.train.Saver().restore(session, checkpoint)
            coord = tf.train.Coordinator(clean_stop_exception_types=(
              tf.errors.CancelledError, tf.errors.OutOfRangeError))
            threads = tf.train.start_queue_runners(coord=coord, sess=session)

            with coord.stop_on_exception():
                tf.logging.info("Validation started for {}".format((checkpoint)))
                while not coord.should_stop():
                    tf.logging.info("Valid loss: {}/{}".format(loss_sum, tot_batch))
                    tf.logging.info("Valid acc for length: {}".format(corr_len/float(tot)))
                    tf.logging.info("Valid acc: {}".format(corr_full/float(tot)))
                    summary = tf.Summary(value=
                        [ tf.Summary.Value(tag="loss_valid", simple_value=loss_sum/tot_batch),
                          tf.Summary.Value(tag="lenghtAcc_valid",
                            simple_value=corr_len/float(tot)),
                          tf.Summary.Value(tag="accuracy_valid",
                            simple_value=corr_full/float(tot))
                          ])
            writer.add_summary(summary, global_step=step)
            writer.flush()
            coord.request_stop()
            coord.join(threads)

def train(
        keep_prob = 0.8,
        max_input_len = 1000,
        max_output_len = 200,
        rnn_size = 256,
        num_layers = 3,
        batch_size = 32,
        learning_rate = 0.0005,
        num_epochs = 5,
        learning_rate_decay = 0.95,
        min_learning_rate = 0.0005,
        display_step = 1, # Check training loss after every display_step batches
        stop_early = 0,
        checkpoint = "gs://wsj-data/checkpoints")

    with tf.Graph().as_default():
        train_op, cost, predictions = get_model()
        self.valid_graph = tf.Graph()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator(clean_stop_exception_types=(
              tf.errors.CancelledError, tf.errors.OutOfRangeError))
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            latest_checkpoint = None

            with coord.stop_on_exception():
              while not coord.should_stop():
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
                latest = tf.train.latest_checkpoint(self.job_dir)
                if not latest == latest_checkpoint:
                  run_eval(latest, eval_graph, step_)
                  latest_checkpoint = latest
            coord.request_stop()
            coord.join(threads)



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-dir', required=True, type=str)
        parser.add_argument('--job-dir', required=True, type=str)
        parser.add_argument('--num-epochs', type=int)
        parse_args, unknown = parser.parse_known_args()
        run(**parse_args.__dict__)
