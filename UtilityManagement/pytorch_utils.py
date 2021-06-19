import torch


def is_gpu_avaliable():
    return torch.cuda.is_available()


def get_gpu_device_name():
    return torch.cuda.get_device_name(0)


def get_gpu_device_count():
    return torch.cuda.device_count()


def set_DataParallel(model):
    return torch.nn.DataParallel(model.features)


def set_cpu(model):
    model.cpu()


def set_gpu(model):
    model.cuda()

