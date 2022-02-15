import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Convolution2D, GlobalAveragePooling2D, UpSampling2D, AveragePooling2D, Dropout
from tensorflow.keras.applications.resnet import ResNet101
from modules.bn import ABN

pretrained_settings = {
    'resnet101': {
        'imagenet': {
            'input_space': 'BGR',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.406, 0.456, 0.485],
            'std': [0.225, 0.224, 0.229],
            'num_classes': 1000
        }
    },
}

class PSPModule(Layer):
    def __init__(self, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.out_feature = out_features
        self.sizes = sizes

        # def _bottleneck(self, feat, filters=512):
        self.conv = Conv2D(filters=512, kernel_size=3, padding='same', dilation_rate=1, use_bias=False)
        self.bn = ABN(slope=0.01)
        

    def _make_stages(self, base, out_features=512):
        # red
        red = GlobalAveragePooling2D(name='red_pool')(base)
        red = tf.keras.layers.Reshape((1, 1, base.shape[3]))(red)
        red = Convolution2D(filters=out_features, kernel_size=(1, 1), name='red_1_by_1')(red)
        red = UpSampling2D(size=16, interpolation='bilinear', name='red_upsampling')(red)
        # yellow
        yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base)
        yellow = Convolution2D(filters=out_features, kernel_size=(1, 1), name='yellow_1_by_1')(yellow)
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
        out = tf.keras.layers.concatenate([base, red, yellow, blue, green], axis=3)
        return out

    def call(self, base):
        encode = self._make_stages(base)
        out = self.conv(encode)
        out = self.bn(out)
        return out

class Edge_Module(tf.keras.layers.Layer):
    def __init__(self, mid_fea = 256, out_fea = 2):
        super(Edge_Module, self).__init__()
        self.conv1 = Conv2D(filters=mid_fea, kernel_size=1, padding='same', strides=(1, 1), use_bias=False)
        self.conv2 = Conv2D(filters=mid_fea, kernel_size=1, padding='same', strides=(1, 1), use_bias=False)
        self.conv3 = Conv2D(filters=mid_fea, kernel_size=1, padding='same', strides=(1, 1), use_bias=False)
        self.conv4 = Conv2D(filters=out_fea, kernel_size=3, padding='same', use_bias=True)
        self.conv5 = Conv2D(filters=out_fea, kernel_size=1, padding='same', use_bias=True)

        self.bn1 = ABN(slope=0.01)
        self.bn2 = ABN(slope=0.01)
        self.bn3 = ABN(slope=0.01)

    def call(self, x1, x2, x3):
        #[b,h,w,c]
        _,h,w,_ = x1.shape

        edge1_fea = self.conv1(x1)
        edge1_fea = self.bn1(edge1_fea)
        edge1 = self.conv4(edge1_fea)

        edge2_fea = self.conv2(x2)
        edge2_fea = self.bn2(edge2_fea)
        edge2 = self.conv4(edge2_fea)

        edge3_fea = self.conv3(x3)
        edge3_fea = self.bn3(edge3_fea)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = tf.compat.v1.image.resize_bilinear(images=edge2_fea, size=[h, w], align_corners=True)
        edge3_fea = tf.compat.v1.image.resize_bilinear(images=edge3_fea, size=[h, w], align_corners=True)
        edge2 = tf.compat.v1.image.resize_bilinear(images=edge2, size=[h, w], align_corners=True)
        edge3 = tf.compat.v1.image.resize_bilinear(images=edge3, size=[h, w], align_corners=True)

        edge = tf.keras.layers.concatenate([edge1, edge2, edge3], axis=3)
        edge_fea = tf.keras.layers.concatenate([edge1_fea, edge2_fea, edge3_fea], axis=3)
        edge = self.conv5(edge)

        return edge, edge_fea

class Decoder_Module(Layer):
    def __init__(self, num_classes):
        super(Decoder_Module, self).__init__()
        self.conv1 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False) #input shape has 512 channels


        self.conv2 = Conv2D(filters=48, kernel_size=1, strides=(1,1), padding='same', use_bias=False) #input shape has 256 channels

        self.conv3 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False) #input shape has 304 channels
        self.conv31 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False) #input shape has 256 channels

        self.conv4 = Conv2D(filters=num_classes, kernel_size=1, padding='same', use_bias=False) #input shape has 256 channels

        self.bn1 = ABN(slope=0.01)
        self.bn2 = ABN(slope=0.01)
        self.bn3 = ABN(slope=0.01)
        self.bn4 = ABN(slope=0.01)


    def call(self, xt, xl):
        h,w = xl.shape[1], xl.shape[2]
        xt = self.conv1(xt)
        xt = self.bn1(xt)
        xt = tf.compat.v1.image.resize_bilinear(xt, size=(h,w), align_corners=True)

        xl = self.conv2(xl)
        xl = self.bn2(xl)
        x = tf.keras.layers.concatenate([xt,xl], axis=3) #[b,h,w,c]

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv31(x)
        x = self.bn4(x)

        seg = self.conv4(x)
        return seg, x

class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.context_encoding = PSPModule() #input: 2048 channels is conv5_act of resnet101, output = 512 channels
        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)

        self.fusion = tf.keras.Sequential([
            Conv2D(filters = 256, kernel_size=1, padding='same', use_bias=False),
            ABN(slope=0.01),
            Dropout(0.1),
            Conv2D(filters=num_classes, kernel_size=1, padding='same', use_bias=True)
        ])

    def call(self,x):
        model =ResNet101(include_top=False, weights='imagenet', input_shape=(512, 512, 3))
        model.trainable = False
        model_to_conv2 = model.get_layer("conv2_block3_out")
        model_to_conv3 = model.get_layer("conv3_block4_out")
        model_to_conv4 = model.get_layer("conv4_block23_out")
        model_to_conv5 = model.get_layer("conv5_block3_out")
        
        new_model = tf.keras.models.Model(inputs = [model.input], outputs = [model_to_conv2.output, model_to_conv3.output, model_to_conv4.output, model_to_conv5.output])
        x2, x3, x4, x5 = new_model(x)

        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x,x2)

        #Edge branch
        edge_result, edge_fea = self.edge(x2, x3, x4)

        #Fusion
        x = tf.keras.layers.concatenate([parsing_fea, edge_fea], axis=3)
        fusion_result = self.fusion(x)
        return [[parsing_result, fusion_result], [edge_result]]

def resnet101(num_classes = 18):
    model = ResNet(num_classes)
    return model