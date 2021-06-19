from ModelManagement.Util.keras_util import *


class MobileNet_V2(tf.keras.models.Model):

    def __init__(self, classes, **kwargs):
        super(MobileNet_V2, self).__init__(**kwargs)

        self.model_name = 'MobileNet_V2'

        expands = (1, 6, 6, 6, 6, 6, 6)
        repeats = (1, 2, 3, 4, 3, 3, 1)
        strides = (1, 2, 2, 2, 1, 2, 1)
        channels = (16, 24, 32, 64, 96, 160, 320)

        output_channel = 32
        self.conv_0 = set_conv(output_channel, kernel=(3, 3), strides=(2, 2), name='conv_0')
        self.bn_0 = set_batch_normalization(name='bn_0')
        # set_relu6

        # ---------------- Inverted Residual Block 1 --------------------- #
        input_channel = output_channel
        self.irb_conv1_1 = set_conv(expands[0] * input_channel, kernel=(1, 1), strides=(strides[0], strides[0]), name='irb_conv1_1')
        self.irb_bn1_1 = set_batch_normalization(name='irb_bn1_1')
        # set_relu6
        self.irb_depth_conv1_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv1_1')
        self.irb_depth_bn1_1 = set_batch_normalization(name='irb_depth_bn1_1')
        # set_relu6
        self.irb_point_conv1_1 = set_conv(channel=channels[0], kernel=(1, 1), name='irb_point_conv1_1')
        self.irb_point_bn1_1 = set_batch_normalization(name='irb_point_bn1_1')
        output_channel = channels[0]

        # ---------------- Inverted Residual Block 2 --> Repeat : 2 --------------------- #
        input_channel = output_channel
        self.irb_conv2_1 = set_conv(expands[1] * input_channel, kernel=(1, 1), strides=(strides[1], strides[1]), name='irb_conv2_1')
        self.irb_bn2_1 = set_batch_normalization(name='irb_bn2_1')
        # set_relu6
        self.irb_depth_conv2_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv2_1')
        self.irb_depth_bn2_1 = set_batch_normalization(name = 'irb_depth_bn2_1')
        # set_relu6
        self.irb_point_conv2_1 = set_conv(channel=channels[1], kernel=(1, 1), name='irb_point_conv2_1')
        self.irb_point_bn2_1 = set_batch_normalization(name='irb_point_bn2_1')
        output_channel = channels[1]

        self.irb_conv2_2 = set_conv(expands[1] * output_channel, kernel=(1, 1), name='irb_conv2_2')
        self.irb_bn2_2 = set_batch_normalization(name='irb_bn2_2')
        # set_relu6
        self.irb_depth_conv2_2 = set_depthwise_conv(kernel=(3, 3) , name='irb_depth_conv2_2')
        self.irb_depth_bn2_2 = set_batch_normalization(name='irb_depth_bn2_2')
        # set_relu6
        self.irb_point_conv2_2 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv2_2')
        self.irb_point_bn2_2 = set_batch_normalization(name='irb_point_bn2_2')
        # net = self.irb_point_bn2_2 + self.irb_point_bn2_1

        # ---------------- Inverted Residual Block 3 --> Repeat : 3 --------------------- #
        input_channel = output_channel
        self.irb_conv3_1 = set_conv(expands[2] * input_channel, kernel=(1, 1), strides=(strides[2], strides[2]), name='irb_conv3_1')
        self.irb_bn3_1 = set_batch_normalization(name='irb_bn3_1')
        # set_relu6
        self.irb_depth_conv3_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv3_1')
        self.irb_depth_bn3_1 = set_batch_normalization(name = 'irb_depth_bn3_1')
        # set_relu6
        self.irb_point_conv3_1 = set_conv(channel=channels[2], kernel=(1, 1), name='irb_point_conv3_1')
        self.irb_point_bn3_1 = set_batch_normalization(name='irb_point_bn3_1')
        output_channel = channels[2]

        self.irb_conv3_2 = set_conv(expands[2] * output_channel, kernel=(1, 1), name='irb_conv3_2')
        self.irb_bn3_2 = set_batch_normalization(name='irb_bn3_2')
        # set_relu6
        self.irb_depth_conv3_2 = set_depthwise_conv(kernel=(3, 3) , name='irb_depth_conv3_2')
        self.irb_depth_bn3_2 = set_batch_normalization(name='irb_depth_bn3_2')
        # set_relu6
        self.irb_point_conv3_2 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv3_2')
        self.irb_point_bn3_2 = set_batch_normalization(name='irb_point_bn3_2')
        # net = self.irb_point_bn3_2 + self.irb_point_bn3_1

        self.irb_conv3_3 = set_conv(expands[2] * output_channel, kernel=(1, 1), name='irb_conv3_3')
        self.irb_bn3_3 = set_batch_normalization(name='irb_bn3_3')
        # set_relu6
        self.irb_depth_conv3_3 = set_depthwise_conv(kernel=(3, 3) , name='irb_depth_conv3_3')
        self.irb_depth_bn3_3 = set_batch_normalization(name='irb_depth_bn3_3')
        # set_relu6
        self.irb_point_conv3_3 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv3_3')
        self.irb_point_bn3_3 = set_batch_normalization(name='irb_point_bn3_3')
        # net = self.irb_point_bn3_3 + self.irb_point_bn3_2

        # ---------------- Inverted Residual Block 4 --> Repeat : 4 --------------------- #
        input_channel = output_channel
        self.irb_conv4_1 = set_conv(expands[3] * input_channel, kernel=(1, 1), strides=(strides[3], strides[3]), name='irb_conv4_1')
        self.irb_bn4_1 = set_batch_normalization(name='irb_bn4_1')
        # set_relu6
        self.irb_depth_conv4_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv4_1')
        self.irb_depth_bn4_1 = set_batch_normalization(name = 'irb_depth_bn4_1')
        # set_relu6
        self.irb_point_conv4_1 = set_conv(channel=channels[3], kernel=(1, 1), name='irb_point_conv4_1')
        self.irb_point_bn4_1 = set_batch_normalization(name='irb_point_bn4_1')
        output_channel = channels[3]

        self.irb_conv4_2 = set_conv(expands[3] * output_channel, kernel=(1, 1), name='irb_conv4_2')
        self.irb_bn4_2 = set_batch_normalization(name='irb_bn4_2')
        # set_relu6
        self.irb_depth_conv4_2 = set_depthwise_conv(kernel=(3, 3) , name='irb_depth_conv4_2')
        self.irb_depth_bn4_2 = set_batch_normalization(name='irb_depth_bn4_2')
        # set_relu6
        self.irb_point_conv4_2 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv4_2')
        self.irb_point_bn4_2 = set_batch_normalization(name='irb_point_bn4_2')
        # net = self.irb_point_bn4_2 + self.irb_point_bn4_1

        self.irb_conv4_3 = set_conv(expands[3] * output_channel, kernel=(1, 1), name='irb_conv4_3')
        self.irb_bn4_3 = set_batch_normalization(name='irb_bn4_3')
        # set_relu6
        self.irb_depth_conv4_3 = set_depthwise_conv(kernel=(3, 3) , name='irb_depth_conv4_3')
        self.irb_depth_bn4_3 = set_batch_normalization(name='irb_depth_bn4_3')
        # set_relu6
        self.irb_point_conv4_3 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv4_3')
        self.irb_point_bn4_3 = set_batch_normalization(name='irb_point_bn4_3')
        # net = self.irb_point_bn4_3 + net(self.irb_point_bn4_2 + self.irb_point_bn4_1)

        self.irb_conv4_4 = set_conv(expands[3] * output_channel, kernel=(1, 1), name='irb_conv4_4')
        self.irb_bn4_4 = set_batch_normalization(name='irb_bn4_4')
        # set_relu6
        self.irb_depth_conv4_4 = set_depthwise_conv(kernel=(3, 3) , name='irb_depth_conv4_4')
        self.irb_depth_bn4_4 = set_batch_normalization(name='irb_depth_bn4_4')
        # set_relu6
        self.irb_point_conv4_4 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv4_4')
        self.irb_point_bn4_4 = set_batch_normalization(name='irb_point_bn4_4')
        # net = self.irb_point_bn4_4 + net(self.irb_point_bn4_3 + net(self.irb_point_bn4_2 + self.irb_point_bn4_1))

        # ---------------- Inverted Residual Block 5 --> Repeat : 3 --------------------- #
        input_channel = output_channel
        self.irb_conv5_1 = set_conv(expands[4] * input_channel, kernel=(1, 1), strides=(strides[4], strides[4]), name='irb_conv5_1')
        self.irb_bn5_1 = set_batch_normalization(name='irb_bn5_1')
        # set_relu6
        self.irb_depth_conv5_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv5_1')
        self.irb_depth_bn5_1 = set_batch_normalization(name = 'irb_depth_bn5_1')
        # set_relu6
        self.irb_point_conv5_1 = set_conv(channel=channels[4], kernel=(1, 1), name='irb_point_conv5_1')
        self.irb_point_bn5_1 = set_batch_normalization(name='irb_point_bn5_1')
        output_channel = channels[4]

        self.irb_conv5_2 = set_conv(expands[4] * output_channel, kernel=(1, 1), name='irb_conv5_2')
        self.irb_bn5_2 = set_batch_normalization(name='irb_bn5_2')
        # set_relu6
        self.irb_depth_conv5_2 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv5_2')
        self.irb_depth_bn5_2 = set_batch_normalization(name='irb_depth_bn5_2')
        # set_relu6
        self.irb_point_conv5_2 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv5_2')
        self.irb_point_bn5_2 = set_batch_normalization(name='irb_point_bn5_2')
        # net = self.irb_point_bn5_2 + self.irb_point_bn5_1

        self.irb_conv5_3 = set_conv(expands[4] * output_channel, kernel=(1, 1), name='irb_conv5_3')
        self.irb_bn5_3 = set_batch_normalization(name='irb_bn5_3')
        # set_relu6
        self.irb_depth_conv5_3 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv5_3')
        self.irb_depth_bn5_3 = set_batch_normalization(name='irb_depth_bn5_3')
        # set_relu6
        self.irb_point_conv5_3 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv5_3')
        self.irb_point_bn5_3 = set_batch_normalization(name='irb_point_bn5_3')
        # net = self.irb_point_bn5_3 + net(self.irb_point_bn5_2 + self.irb_point_bn5_1)

        # ---------------- Inverted Residual Block 6 --> Repeat : 3 --------------------- #
        input_channel = output_channel
        self.irb_conv6_1 = set_conv(expands[5] * input_channel, kernel=(1, 1), strides=(strides[5], strides[5]), name='irb_conv6_1')
        self.irb_bn6_1 = set_batch_normalization(name='irb_bn6_1')
        # set_relu6
        self.irb_depth_conv6_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv6_1')
        self.irb_depth_bn6_1 = set_batch_normalization(name = 'irb_depth_bn6_1')
        # set_relu6
        self.irb_point_conv6_1 = set_conv(channel=channels[5], kernel=(1, 1), name='irb_point_conv6_1')
        self.irb_point_bn6_1 = set_batch_normalization(name='irb_point_bn6_1')
        output_channel = channels[5]

        self.irb_conv6_2 = set_conv(expands[5] * output_channel, kernel=(1, 1), name='irb_conv6_2')
        self.irb_bn6_2 = set_batch_normalization(name='irb_bn6_2')
        # set_relu6
        self.irb_depth_conv6_2 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv6_2')
        self.irb_depth_bn6_2 = set_batch_normalization(name='irb_depth_bn6_2')
        # set_relu6
        self.irb_point_conv6_2 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv6_2')
        self.irb_point_bn6_2 = set_batch_normalization(name='irb_point_bn6_2')
        # net = self.irb_point_bn6_2 + self.irb_point_bn6_1

        self.irb_conv6_3 = set_conv(expands[5] * output_channel, kernel=(1, 1), name='irb_conv6_3')
        self.irb_bn6_3 = set_batch_normalization(name='irb_bn6_3')
        # set_relu6
        self.irb_depth_conv6_3 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv6_3')
        self.irb_depth_bn6_3 = set_batch_normalization(name='irb_depth_bn6_3')
        # set_relu6
        self.irb_point_conv6_3 = set_conv(channel=output_channel, kernel=(1, 1), name='irb_point_conv6_3')
        self.irb_point_bn6_3 = set_batch_normalization(name='irb_point_bn6_3')
        # net = self.irb_point_bn6_3 + net(self.irb_point_bn6_2 + self.irb_point_bn6_1)

        # ---------------- Inverted Residual Block 7 --------------------- #
        input_channel = output_channel
        self.irb_conv7_1 = set_conv(expands[6] * input_channel, kernel=(1, 1), strides=(strides[6], strides[6]), name='irb_conv7_1')
        self.irb_bn7_1 = set_batch_normalization(name='irb_bn7_1')
        # set_relu6
        self.irb_depth_conv7_1 = set_depthwise_conv(kernel=(3, 3), name='irb_depth_conv7_1')
        self.irb_depth_bn7_1 = set_batch_normalization(name = 'irb_depth_bn7_1')
        # set_relu6
        self.irb_point_conv7_1 = set_conv(channel=channels[6], kernel=(1, 1), name='irb_point_conv7_1')
        self.irb_point_bn7_1 = set_batch_normalization(name='irb_point_bn7_1')

        output_channel = 1280
        self.conv_8 = set_conv(output_channel, kernel=(1, 1), strides=(1, 1), name='conv_8')
        self.bn_8 = set_batch_normalization(name='bn_8')
        # set_relu6

        self.gap = set_global_average_pooling()
        self.fcl = set_dense(channel=classes, name='fcl', activation='softmax')


    def call(self, inputs, training):
        net = self.conv_0(inputs)
        net = self.bn_0(net, training)
        net = set_relu6(net)

        # ---------------- Inverted Residual Block 1 --------------------- #
        net = self.irb_conv1_1(net)
        net = self.irb_bn1_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv1_1(net)
        net = self.irb_depth_bn1_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv1_1(net)
        net = self.irb_point_bn1_1(net, training)

        # ---------------- Inverted Residual Block 2 --> Repeat : 2 --------------------- #
        net = self.irb_conv2_1(net)
        net = self.irb_bn2_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv2_1(net)
        net = self.irb_depth_bn2_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv2_1(net)
        irb2_1 = self.irb_point_bn2_1(net, training)

        net = self.irb_conv2_2(irb2_1)
        net = self.irb_bn2_2(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv2_2(net)
        net = self.irb_depth_bn2_2(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv2_2(net)
        irb2_2 = self.irb_point_bn2_2(net, training)

        net = irb2_1 + irb2_2

        # ---------------- Inverted Residual Block 3 --> Repeat : 3 --------------------- #
        net = self.irb_conv3_1(net)
        net = self.irb_bn3_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv3_1(net)
        net = self.irb_depth_bn3_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv3_1(net)
        irb3_1 = self.irb_point_bn3_1(net, training)

        net = self.irb_conv3_2(irb3_1)
        net = self.irb_bn3_2(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv3_2(net)
        net = self.irb_depth_bn3_2(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv3_2(net)
        irb3_2 = self.irb_point_bn3_2(net, training)

        add_irb3 = irb3_1 + irb3_2

        net = self.irb_conv3_3(add_irb3)
        net = self.irb_bn3_3(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv3_3(net)
        net = self.irb_depth_bn3_3(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv3_3(net)
        irb3_3 = self.irb_point_bn3_3(net, training)

        net = add_irb3 + irb3_3

        # ---------------- Inverted Residual Block 4 --> Repeat : 4 --------------------- #
        net = self.irb_conv4_1(net)
        net = self.irb_bn4_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv4_1(net)
        net = self.irb_depth_bn4_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv4_1(net)
        irb4_1 = self.irb_point_bn4_1(net, training)

        net = self.irb_conv4_2(irb4_1)
        net = self.irb_bn4_2(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv4_2(net)
        net = self.irb_depth_bn4_2(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv4_2(net)
        irb4_2 = self.irb_point_bn4_2(net, training)

        add_irb4_1 = irb4_1 + irb4_2

        net = self.irb_conv4_3(add_irb4_1)
        net = self.irb_bn4_3(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv4_3(net)
        net = self.irb_depth_bn4_3(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv4_3(net)
        irb4_3 = self.irb_point_bn4_3(net, training)

        add_irb4_2 = add_irb4_1 + irb4_3

        net = self.irb_conv4_4(add_irb4_2)
        net = self.irb_bn4_4(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv4_4(net)
        net = self.irb_depth_bn4_4(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv4_4(net)
        irb4_4 = self.irb_point_bn4_4(net, training)

        net = add_irb4_2 + irb4_4

        # ---------------- Inverted Residual Block 5 --> Repeat : 3 --------------------- #
        net = self.irb_conv5_1(net)
        net = self.irb_bn5_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv5_1(net)
        net = self.irb_depth_bn5_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv5_1(net)
        irb5_1 = self.irb_point_bn5_1(net, training)

        net = self.irb_conv5_2(irb5_1)
        net = self.irb_bn5_2(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv5_2(net)
        net = self.irb_depth_bn5_2(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv5_2(net)
        irb5_2 = self.irb_point_bn5_2(net, training)

        add_irb5 = irb5_1 + irb5_2

        net = self.irb_conv5_3(add_irb5)
        net = self.irb_bn5_3(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv5_3(net)
        net = self.irb_depth_bn5_3(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv5_3(net)
        irb5_3 = self.irb_point_bn5_3(net, training)

        net = add_irb5 + irb5_3

        # ---------------- Inverted Residual Block 6 --> Repeat : 3 --------------------- #
        net = self.irb_conv6_1(net)
        net = self.irb_bn6_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv6_1(net)
        net = self.irb_depth_bn6_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv6_1(net)
        irb6_1 = self.irb_point_bn6_1(net, training)

        net = self.irb_conv6_2(irb6_1)
        net = self.irb_bn6_2(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv6_2(net)
        net = self.irb_depth_bn6_2(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv6_2(net)
        irb6_2 = self.irb_point_bn6_2(net, training)

        add_irb6 = irb6_1 + irb6_2

        net = self.irb_conv6_3(add_irb6)
        net = self.irb_bn6_3(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv6_3(net)
        net = self.irb_depth_bn6_3(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv6_3(net)
        irb6_3 = self.irb_point_bn6_3(net, training)

        net = add_irb6 + irb6_3

        # ---------------- Inverted Residual Block 7 --------------------- #
        net = self.irb_conv7_1(net)
        net = self.irb_bn7_1(net, training)
        net = set_relu6(net)
        net = self.irb_depth_conv7_1(net)
        net = self.irb_depth_bn7_1(net, training)
        net = set_relu6(net)
        net = self.irb_point_conv7_1(net)
        net = self.irb_point_bn7_1(net, training)

        net = self.conv_8(net)
        net = self.bn_8(net, training)
        net = set_relu6(net)

        net = self.gap(net)
        net = self.fcl(net)
        return net

    def get_name(self):
        return self.model_name
