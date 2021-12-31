from ModelManagement.Util.pytorch_util import *
import os.path
import math
import warnings


class MobileNet_V1(nn.Module):
    def __init__(self, classes, first_channel=32, **kwargs):
        super(MobileNet_V1, self).__init__(**kwargs)

        self.model_name = 'MobileNet_V1'
        in_channel = 3
        channels = (
            first_channel, first_channel * 2, first_channel * 4, first_channel * 8, first_channel * 16,
            first_channel * 32)

        layer_list = []

        conv_0 = set_conv(in_channel, channels[0], kernel=3, strides=2, padding=1)
        layer_list += [conv_0, set_batch_normalization(channels[0]), set_relu(True)]

        depth_conv_1 = set_detphwise_conv(channels[0], channels[0])
        layer_list += [depth_conv_1, set_batch_normalization(channels[0]), set_relu(True)]
        point_conv_1 = set_pointwise_conv(channels[0], channels[1], kernel=1)
        layer_list += [point_conv_1, set_batch_normalization(channels[1]), set_relu(True)]

        depth_conv_2 = set_detphwise_conv(channels[1], channels[1], strides=2)
        layer_list += [depth_conv_2, set_batch_normalization(channels[1]), set_relu(True)]
        point_conv_2 = set_pointwise_conv(channels[1], channels[2], kernel=1)
        layer_list += [point_conv_2, set_batch_normalization(channels[2]), set_relu(True)]
        depth_conv_3 = set_detphwise_conv(channels[2], channels[2], strides=1)
        layer_list += [depth_conv_3, set_batch_normalization(channels[2]), set_relu(True)]
        point_conv_3 = set_pointwise_conv(channels[2], channels[2], kernel=1)
        layer_list += [point_conv_3, set_batch_normalization(channels[2]), set_relu(True)]

        depth_conv_4 = set_detphwise_conv(channels[2], channels[2], strides=2)
        layer_list += [depth_conv_4, set_batch_normalization(channels[2]), set_relu(True)]
        point_conv_4 = set_pointwise_conv(channels[2], channels[3], kernel=1)
        layer_list += [point_conv_4, set_batch_normalization(channels[3]), set_relu(True)]
        depth_conv_5 = set_detphwise_conv(channels[3], channels[3], strides=1)
        layer_list += [depth_conv_5, set_batch_normalization(channels[3]), set_relu(True)]
        point_conv_5 = set_pointwise_conv(channels[3], channels[3], kernel=1)
        layer_list += [point_conv_5, set_batch_normalization(channels[3]), set_relu(True)]

        depth_conv_6 = set_detphwise_conv(channels[3], channels[3], strides=2)
        layer_list += [depth_conv_6, set_batch_normalization(channels[3]), set_relu(True)]
        point_conv_6 = set_pointwise_conv(channels[3], channels[4], kernel=1)
        layer_list += [point_conv_6, set_batch_normalization(channels[4]), set_relu(True)]
        depth_conv_7 = set_detphwise_conv(channels[4], channels[4], strides=1)
        layer_list += [depth_conv_7, set_batch_normalization(channels[4]), set_relu(True)]
        point_conv_7 = set_pointwise_conv(channels[4], channels[4], kernel=1)
        layer_list += [point_conv_7, set_batch_normalization(channels[4]), set_relu(True)]
        depth_conv_8 = set_detphwise_conv(channels[4], channels[4], strides=1)
        layer_list += [depth_conv_8, set_batch_normalization(channels[4]), set_relu(True)]
        point_conv_8 = set_pointwise_conv(channels[4], channels[4], kernel=1)
        layer_list += [point_conv_8, set_batch_normalization(channels[4]), set_relu(True)]
        depth_conv_9 = set_detphwise_conv(channels[4], channels[4], strides=1)
        layer_list += [depth_conv_9, set_batch_normalization(channels[4]), set_relu(True)]
        point_conv_9 = set_pointwise_conv(channels[4], channels[4], kernel=1)
        layer_list += [point_conv_9, set_batch_normalization(channels[4]), set_relu(True)]
        depth_conv_10 = set_detphwise_conv(channels[4], channels[4], strides=1)
        layer_list += [depth_conv_10, set_batch_normalization(channels[4]), set_relu(True)]
        point_conv_10 = set_pointwise_conv(channels[4], channels[4], kernel=1)
        layer_list += [point_conv_10, set_batch_normalization(channels[4]), set_relu(True)]
        depth_conv_11 = set_detphwise_conv(channels[4], channels[4], strides=1)
        layer_list += [depth_conv_11, set_batch_normalization(channels[4]), set_relu(True)]
        point_conv_11 = set_pointwise_conv(channels[4], channels[4], kernel=1)
        layer_list += [point_conv_11, set_batch_normalization(channels[4]), set_relu(True)]

        depth_conv_12 = set_detphwise_conv(channels[4], channels[4], strides=2)
        layer_list += [depth_conv_12, set_batch_normalization(channels[4]), set_relu(True)]
        point_conv_12 = set_pointwise_conv(channels[4], channels[5], kernel=1)
        layer_list += [point_conv_12, set_batch_normalization(channels[5]), set_relu(True)]
        depth_conv_13 = set_detphwise_conv(channels[5], channels[5], strides=1)
        layer_list += [depth_conv_13, set_batch_normalization(channels[5]), set_relu(True)]
        point_conv_13 = set_pointwise_conv(channels[5], channels[5], kernel=1)
        layer_list += [point_conv_13, set_batch_normalization(channels[5]), set_relu(True)]

        self.gap = set_global_average_pooling()
        self.fcl = set_dense(channels[5], classes)

        self.features = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fcl(x)
        return x

    def get_name(self):
        return self.model_name

    def initialize_weights(self, init_weights):
        if init_weights is True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                    if m.bias is not None:
                        m.bias.data.zero_()

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


def MobileNet_v1(classes, first_channel):
    pretrained_path ="./Log/"
    model = MobileNet_V1(classes, first_channel)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model

