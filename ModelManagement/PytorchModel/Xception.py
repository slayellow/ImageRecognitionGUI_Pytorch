from ModelManagement.Util.pytorch_util import *
import os.path
import math
import torch.nn.functional as F


# class SeparableConv2d(nn.Module):
#     def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1):
#         super(SeparableConv2d,self).__init__()
#
#         self.conv1 = set_detphwise_conv(in_channels, in_channels, kernel=kernel_size, strides=stride, padding=padding, dilation=dilation)
#         self.pointwise = set_pointwise_conv(in_channels, out_channels, kernel=1, strides=1, padding=0, dilation=1)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.pointwise(x)
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
#         super(Block, self).__init__()
#
#         if out_filters != in_filters or strides!=1:
#             self.skip = set_conv(in_filters, out_filters, kernel=1, strides=strides, padding=0)
#             self.skipbn = set_batch_normalization(out_filters)
#         else:
#             self.skip=None
#
#         rep=[]
#
#         filters=in_filters
#         if grow_first:
#             rep.append(set_relu())
#             rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1))
#             rep.append(set_batch_normalization(out_filters))
#             filters = out_filters
#
#         for i in range(reps-1):
#             rep.append(set_relu())
#             rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1))
#             rep.append(set_batch_normalization(filters))
#
#         if not grow_first:
#             rep.append(set_relu())
#             rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1))
#             rep.append(set_batch_normalization(out_filters))
#
#         if not start_with_relu:
#             rep = rep[1:]
#         else:
#             rep[0] = set_relu()
#
#         if strides != 1:
#             rep.append(set_max_pool(kernel=3,strides=strides,padding=1))
#         self.rep = nn.Sequential(*rep)
#
#     def forward(self,inp):
#         x = self.rep(inp)
#
#         if self.skip is not None:
#             skip = self.skip(inp)
#             skip = self.skipbn(skip)
#         else:
#             skip = inp
#
#         x+=skip
#         return x
#
#
# class Xception(nn.Module):
#
#     def __init__(self, classes, **kwargs):
#         super(Xception, self).__init__(**kwargs)
#
#         self.model_name = 'Xception'
#         in_channel = 3
#
#         self.conv1 = set_conv(3, 32, kernel=3, strides=2, padding=1)
#         self.bn1 = set_batch_normalization(32)
#         self.relu1 = set_relu()
#
#         self.conv2 = set_conv(32, 64, kernel=3, strides=1, padding=1)
#         self.bn2 = set_batch_normalization(64)
#         self.relu2 = set_relu()
#
#         self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
#         self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
#         self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
#
#         self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#
#         self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#         self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
#
#         self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
#
#         self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
#         self.bn3 = set_batch_normalization(1536)
#         self.relu3 = set_relu()
#
#         # do relu here
#         self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
#         self.bn4 = set_batch_normalization(2048)
#         self.relu4 = set_relu()
#
#         self.gap = set_global_average_pooling()
#
#         self.fcl = set_dense(2048, classes)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         x = self.block5(x)
#         x = self.block6(x)
#         x = self.block7(x)
#         x = self.block8(x)
#         x = self.block9(x)
#         x = self.block10(x)
#         x = self.block11(x)
#         x = self.block12(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x)
#
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.relu4(x)
#
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         x = self.fcl(x)
#
#         return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes
        self.model_name = 'Xception'
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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


def load_Xception(classes):
    pretrained_path ="./Log/"
    model = Xception(classes)

    if os.path.isfile(os.path.join(pretrained_path, model.get_name()+'.pth')):
        model.initialize_weights(init_weights=False)
        checkpoint = load_weight_file(os.path.join(pretrained_path, model.get_name() + '.pth'))
        load_weight_parameter(model, checkpoint['state_dict'])
    else:
        model.initialize_weights(init_weights=True)

    return model

