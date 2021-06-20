from ModelManagement.Util.pytorch_util import *
import os.path
import math


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6):
        super().__init__()

        self.residual = nn.Sequential(
            set_conv(in_channels, in_channels * t, kernel=1, padding=0),
            set_batch_normalization(in_channels * t),
            set_relu6(True),

            set_detphwise_conv(in_channels * t, in_channels * t, kernel=3, strides=stride, padding=1),
            set_batch_normalization(in_channels * t),
            set_relu6(True),

            set_conv(in_channels * t, out_channels, kernel=1, padding=0),
            set_batch_normalization(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


class MobileNet_V2(nn.Module):

    def __init__(self, classes=1000):
        super().__init__()

        self.model_name = 'MobileNet_V2'

        self.pre = nn.Sequential(
            set_conv(3, 32, kernel=3, strides=2, padding=1),
            set_batch_normalization(32),
            set_relu6(True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            set_conv(320, 1280, kernel=1, padding=0),
            set_batch_normalization(1280),
            set_relu6(True)
        )
        self.gap = set_global_average_pooling()
        self.conv2 = set_conv(1280, classes, kernel=1, padding=0)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = self.gap(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def get_name(self):
        return self.model_name

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layer_list = []
        layer_list.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layer_list.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layer_list)

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

def MobileNet_v2(classes):
    pretrained_path ="./Log/"
    model = MobileNet_V2(classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model

