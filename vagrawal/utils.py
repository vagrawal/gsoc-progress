import tensorflow as tf

# A custom class inheriting tf.gfile.Open for providing seek with whence
class FileOpen(tf.gfile.Open):
    def seek(self, position, whence = 0):
        if (whence == 0):
            tf.gfile.Open.seek(self, position)
        elif (whence == 1):
            tf.gfile.Open.seek(self, self.tell() + position)
        else:
            raise FileError

