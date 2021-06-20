import torch.nn as nn
import torch
import pytorch_model_summary


def set_conv(in_channel, out_channel, kernel=3, strides=1, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=strides, padding=padding)


def set_detphwise_conv(in_channel, out_channel, kernel=3, strides=1, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=strides, padding=padding, groups=in_channel)


def set_pointwise_conv(in_channel, out_channel, kernel, strides=1, padding=0):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=strides, padding=padding)


def set_batch_normalization(channel):
    return nn.BatchNorm2d(channel)


def set_relu(use_input=True):
    return nn.ReLU(inplace=use_input)


def set_relu6(use_input=True):
    return nn.ReLU6(inplace=use_input)


def set_avg_pool(kernel, strides=1):
    return nn.AvgPool2d(kernel_size=kernel, stride=strides)


def set_max_pool(kernel, strides=2, padding=0):
    return nn.MaxPool2d(kernel_size=kernel, stride=strides, padding=padding)


def set_dense(in_channel, out_channel):
    return nn.Linear(in_channel, out_channel)


def summary(model):
    return pytorch_model_summary.summary(model, torch.zeros(1, 3, 224, 224), show_input=True)


def set_dropout(rate=0.5):
    return nn.Dropout(p=rate)


def set_global_average_pooling():
    return nn.AdaptiveAvgPool2d(1)


def save_weight_parameter(model, name, ext='h5'):
    if ext == 'h5':
        model.save_weights(name + '.' + ext, save_format=ext)


def load_weight_parameter(model, name):
    model.load_state_dict(name)


def load_weight_file(file):
    return torch.load(file)


def loss_cross_entropy():
    return nn.CrossEntropyLoss()


def set_SGD(model, learning_rate, momentum=0.9, weight_decay=1e-4):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def set_Adam(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def set_Adagrad(model, learning_rate):
    return torch.optim.Adagrad(model.parameters(), lr=learning_rate)


def set_RMSProp(model, learning_rate):
    return torch.optim.RMSprop(model.parameters(), lr=learning_rate)