import tensorflow as tf
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy

class CrossEntropy(Loss):
    def __init__(self, ignored_index = 255):
        super(CrossEntropy, self).__init__()
        self.ignored_index = ignored_index
        self.entropy = SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        """
        Cross entropy loss function with ignore index
        :param y_true -- labels -- b, h, w
        :param y_pred -- probability tensor -- b, h, w, c
        :return:
        """
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        C = y_pred.shape[3]
        y_pred = tf.reshape(y_pred, (-1, C))
        y_true = tf.reshape(y_true, (-1,))

        valid = tf.not_equal(y_true, self.ignored_index)
        vy_pred = tf.boolean_mask(y_pred, valid)
        vy_true = tf.boolean_mask(y_true, valid)

        loss = self.entropy(vy_true, vy_pred)

        return loss