from ModelManagement.Util.keras_util import *
from ModelManagement.Util.basic_block import BasicBlock
from ModelManagement.Util.bottleneck_block import BottleneckBlock

import warnings


class ResNet(tf.keras.models.Model):
    # ResNet-18     : BasicBlock        :   [2, 2, 2, 2]
    # ResNet-34     : BasicBlock        :   [3, 4, 6, 3]
    # ResNet-50     : BottleNeckBlock   :   [3, 4, 6, 3]
    # ResNet-101    : BottleNeckBlock   :   [3, 4, 23, 3]
    # ResNet-152    : BottleNeckBlock   :   [3, 4, 36, 3]
    # Channel       : [64, 128, 256, 512]
    def __init__(self, layer_num, classes, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.model_name = 'ResNet_{}'.format(layer_num)

        # ResNet의 기본 구성
        blocks = {18: (2, 2, 2, 2),
                  34: (3, 4, 6, 3),
                  50: (3, 4, 6, 3),
                  101: (3, 4, 23, 3),
                  152: (3, 4, 36, 3)}
        channels = (64, 128, 256, 512)

        if layer_num is 18 or layer_num is 34:
            self.block = BasicBlock
        elif layer_num is 50 or layer_num is 101 or layer_num is 152:
            self.block = BottleneckBlock
        else:
            warnings.warn("클래스가 구성하는 Layer 갯수와 맞지 않습니다.")

        self.conv0 = set_conv(channels[0], (7, 7), strides=(2, 2), name='conv0')

        self.block_list = []
        for idx, (block, channel) in enumerate(zip(blocks[layer_num], channels), start=1):
            if idx == 1:
                if layer_num is 18 or layer_num is 34:
                    self.block_list.append(self.block(channel, name='conv1_0'))
                else:
                    self.block_list.append(self.block(channel, first=True, name='conv1_0'))
            else:
                self.block_list.append(self.block(channel, strides=(2, 2), name='conv{}_0'.format(idx)))

            for block_idx in range(1, block):
                self.block_list.append(self.block(channel, name='conv{}_{}'.format(idx, block_idx)))

        self.bn = set_batch_normalization(name='bn')
        self.gap = set_global_average_pooling()
        self.fcl = set_dense(classes, name='fcl', activation='softmax')

    def call(self, inputs, training):
        net = self.conv0(inputs)
        net = set_max_pool(net, ksize=(3, 3))

        for block in self.block_list:
            net = block(net, training)

        net = self.bn(net, training)
        net = set_relu(net)
        net = self.gap(net)
        net = self.fcl(net)
        return net

    def get_name(self):
        return self.model_name
