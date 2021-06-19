from ModelManagement.Util.pytorch_util import *

import os.path
import math
import warnings


class VGGNet(nn.Module):
    # VGGNet-16     : Block             :   [2, 2, 3, 3, 3]
    # VGGNet-19     : Block             :   [2, 2, 4, 4, 4]

    # Channel       : [64, 128, 256, 512, 512]
    def __init__(self, layer_num, classes):
        super(VGGNet, self).__init__()

        self.model_name = 'VGGNet_{}'.format(layer_num)
        self.layer_num = layer_num

        channels = (64, 128, 256, 512, 512)

        layer_list = []
        in_channel = 3

        if layer_num is 16 or layer_num is 19:

            # Layer 1
            conv0_0 = set_conv(in_channel, channels[0], kernel=3, padding=1)
            layer_list += [conv0_0, set_batch_normalization(channels[0]), set_relu(use_input=True), set_dropout(0.5)]

            conv0_1 = set_conv(channels[0], channels[0], kernel=3, padding=1)
            layer_list += [conv0_1, set_batch_normalization(channels[0]), set_relu(use_input=True), set_max_pool(2, 2)]

            # Layer 2
            conv1_0 = set_conv(channels[0], channels[1], kernel=3, padding=1)
            layer_list += [conv1_0, set_batch_normalization(channels[1]), set_relu(use_input=True), set_dropout(0.5)]

            conv1_1 = set_conv(channels[1], channels[1], kernel=3, padding=1)
            layer_list += [conv1_1, set_batch_normalization(channels[1]), set_relu(use_input=True), set_max_pool(2, 2)]

            # Layer 3
            conv2_0 = set_conv(channels[1], channels[2], kernel=3, padding=1)
            layer_list += [conv2_0, set_batch_normalization(channels[2]), set_relu(use_input=True), set_dropout(0.5)]

            conv2_1 = set_conv(channels[2], channels[2], kernel=3, padding=1)
            layer_list += [conv2_1, set_batch_normalization(channels[2]), set_relu(use_input=True), set_dropout(0.5)]

            conv2_2 = set_conv(channels[2], channels[2], kernel=3, padding=1)
            layer_list += [conv2_2, set_batch_normalization(channels[2]), set_relu(use_input=True)]

            if layer_num is 19:
                conv2_3 = set_conv(channels[2], channels[2], kernel=3, padding=1)
                layer_list += [conv2_3, set_batch_normalization(channels[2]), set_relu(use_input=True)]

            layer_list += [set_max_pool(2, 2)]

            # Layer 4
            conv3_0 = set_conv(channels[2], channels[3], kernel=3, padding=1)
            layer_list += [conv3_0, set_batch_normalization(channels[3]), set_relu(use_input=True), set_dropout(0.5)]

            conv3_1 = set_conv(channels[3], channels[3], kernel=3, padding=1)
            layer_list += [conv3_1, set_batch_normalization(channels[3]), set_relu(use_input=True), set_dropout(0.5)]

            conv3_2 = set_conv(channels[3], channels[3], kernel=3, padding=1)
            layer_list += [conv3_2, set_batch_normalization(channels[3]), set_relu(use_input=True)]

            if layer_num is 19:
                conv3_3 = set_conv(channels[3], channels[3], kernel=3, padding=1)
                layer_list += [conv3_3, set_batch_normalization(channels[3]), set_relu(use_input=True)]

            layer_list += [set_max_pool(2, 2)]

            # Layer 5
            conv4_0 = set_conv(channels[3], channels[4], kernel=3, padding=1)
            layer_list += [conv4_0, set_batch_normalization(channels[4]), set_relu(use_input=True), set_dropout(0.5)]

            conv4_1 = set_conv(channels[4], channels[4], kernel=3, padding=1)
            layer_list += [conv4_1, set_batch_normalization(channels[4]), set_relu(use_input=True), set_dropout(0.5)]

            conv4_2 = set_conv(channels[4], channels[4], kernel=3, padding=1)
            layer_list += [conv4_2, set_batch_normalization(channels[4]), set_relu(use_input=True)]

            if layer_num is 19:
                conv4_3 = set_conv(channels[4], channels[4], kernel=3, padding=1)
                layer_list += [conv4_3, set_batch_normalization(channels[4]), set_relu(use_input=True)]

            layer_list += [set_max_pool(2, 2)]

        else:
            warnings.warn("클래스가 구성하는 Layer 갯수와 맞지 않습니다.")

        # VGG Block
        self.features = nn.Sequential(*layer_list)

        # VGG Block 구성 이후 Layer
        self.classifier = nn.Sequential(
            set_dense(7 * 7 * channels[4], 4096),
            set_relu(use_input=True),
            set_dropout(),
            set_dense(4096, 4096),
            set_relu(use_input=True),
            set_dropout(),
            set_dense(4096, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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


def VGG16(layer_num, classes):
    pretrained_path ="./Log/"
    model = VGGNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model


def VGG19(layer_num, classes):
    pretrained_path ="./Log/"
    model = VGGNet(layer_num, classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)


    return model



