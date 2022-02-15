import tensorflow as tf


class ABN(tf.keras.layers.Layer):
    def __init__(self, momentum=0.99, eps=1e-5, slope = 0.01):
        super(ABN, self).__init__()
        self.momentum = momentum
        self.eps = eps
        self.slope = slope
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum, epsilon=self.eps)
        self.lrelu = tf.keras.layers.LeakyReLU(alpha = self.slope)

    def call(self, x):
        x = self.bn(x)
        x = self.lrelu(x)
        return x


