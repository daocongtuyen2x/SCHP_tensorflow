import tensorflow as tf

class PSPModule(tf.keras.layers.Layer):
    def __init__(self, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

    def _bottleneck(self, feat,  filters=512):
        x = tf.keras.layers.Conv2D(filter = filters, kernel_size=3, padding='same', dilation_rate=1, use_bias=False)(feat)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x
        
    
    def _make_stages(self, base , out_features = 512):

        #red
        red = GlobalAveragePooling2D(name='red_pool')(base)
        red = tf.keras.layers.Reshape((1, 1, 256))(red)
        red = Convolution2D(filters=out_features, kernel_size=(1, 1), name='red_1_by_1')(red)
        red = UpSampling2D(size=256, interpolation='bilinear', name='red_upsampling')(red)
        # yellow
        yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base)
        yellow = Convolution2D(filter= out_features, kernel_size=(1, 1), name='yellow_1_by_1')(yellow)
        yellow = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow)
        # blue
        blue = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base)
        blue = Convolution2D(filters=out_features, kernel_size=(1, 1), name='blue_1_by_1')(blue)
        blue = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue)
        # green
        green = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base)
        green = Convolution2D(filters=out_features, kernel_size=(1, 1), name='green_1_by_1')(green)
        green = UpSampling2D(size=8, interpolation='bilinear', name='green_upsampling')(green)
        # base + red + yellow + blue + green
        return tf.keras.layers.concatenate([base, red, yellow, blue, green])

    def call(self, base):
        encode = _make_stages(base)
        out = _bottleneck(encode)
        return out







