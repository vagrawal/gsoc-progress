import numpy as np
from python_speech_features.base import fbank, delta
import scipy.io.wavfile
import tensorflow as tf
import threading
import random

vocab = np.asarray(
        ['<eps>', '<s>', '</s>'] + list(" '.-ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['<backoff>'])
vocab_to_int = {}

for ch in vocab:
    vocab_to_int[ch] = len(vocab_to_int)

# backoff is not in vocabulary
vocab_size = len(vocab) - 1

# A custom class inheriting tf.gfile.Open for providing seek with whence
class FileOpen(tf.gfile.Open):
    def seek(self, position, whence = 0):
        if (whence == 0):
            tf.gfile.Open.seek(self, position)
        elif (whence == 1):
            tf.gfile.Open.seek(self, self.tell() + position)
        else:
            raise FileError

def get_features(file, nfilt):
    sample_rate, signal = scipy.io.wavfile.read(FileOpen(file))
    feat, energy = fbank(signal, sample_rate, nfilt=nfilt)
    feat = np.log(feat)
    dfeat = delta(feat, 2)
    ddfeat = delta(dfeat, 2)
    return np.concatenate([feat, dfeat, ddfeat, np.expand_dims(energy, 1)], axis=1)

def get_speaker_stats(root, nfilt, set_ids):
    tf.logging.info("Getting speaker stats")
    trans = FileOpen(root + 'transcripts/wsj0/wsj0.trans').readlines()
    sum = {}
    sum_sq = {}
    count = {}
    for line in trans:
        _, file = line.split('(')
        file = file[:-2]
        if (file.split('/')[2] in set_ids and 'wv1' == file.split('.')[1]):
            speaker = file.split('/')[3]
            n_feat = 3 * nfilt + 1
            if speaker not in sum:
                sum[speaker] = np.zeros(n_feat)
                sum_sq[speaker] = np.zeros(n_feat)
                count[speaker] = 0
            feat = get_features(root + 'wav/' + file, nfilt)
            sum[speaker] += np.mean(feat, 0)
            sum_sq[speaker] += np.mean(np.square(feat), 0)
            count[speaker] += 1
    mean =  {k: sum[k] / count[k] for k, v in sum.items()}
    var =  {k: sum_sq[k] / count[k] - np.square(mean[k]) for k, v in sum.items()}
    return mean, var

def read_data_queue(set_id, queue, root, nfilt, sess,
        mean_speaker, var_speaker):
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, nfilt * 3 + 1])
    input_length = tf.placeholder(dtype=tf.int32, shape=[])
    output_data = tf.placeholder(dtype=tf.int32, shape=[None])
    output_length =  tf.placeholder(dtype=tf.int32, shape=[])
    enqueue_op = queue.enqueue(
            [input_data, input_length, output_data, output_length])
    close_op = queue.close()

    thread = threading.Thread(
        target=read_data_thread,
        args=(
            set_id,
            root,
            nfilt,
            queue,
            sess,
            input_data,
            input_length,
            output_data,
            output_length,
            enqueue_op,
            close_op,
            mean_speaker,
            var_speaker))
    thread.daemon = True  # Thread will close when parent quits.
    thread.start()

def read_data_thread(
        set_id,
        root,
        nfilt,
        sess,
        input_data,
        input_length,
        output_data,
        output_length,
        enqueue_op,
        close_op,
        mean_speaker,
        var_speaker):
    trans = FileOpen(root + 'transcripts/wsj0/wsj0.trans').readlines()
    random.shuffle(trans)
    for line in trans:
        text, file = line.split('(')
        text = text[:-1]
        # Remove sounds
        text = "".join(text.split("++")[::2])
        text = [vocab_to_int[c] for c in list(text)] + [vocab_to_int['<EOS>']]
        file = file[:-2]
        if (set_id == file.split('/')[2] and 'wv1' == file.split('.')[1]):
            feat = get_features(root + 'wav/' + file, nfilt)
            feat = feat - mean_speaker[file.split('/')[3]]
            feat = feat / np.sqrt(var_speaker[file.split('/')[3]])
            sess.run(enqueue_op, feed_dict={
                input_data: feat,
                input_length: feat.shape[0],
                output_data: text,
                output_length:len(text)})
    sess.run(close_op)

