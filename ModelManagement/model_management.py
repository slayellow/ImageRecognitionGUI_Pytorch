from ModelManagement.PytorchModel.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from ModelManagement.PytorchModel.VGGNet import VGG16, VGG19
from ModelManagement.PytorchModel.MobileNet_V1 import MobileNet_v1
from ModelManagement.PytorchModel.MobileNet_V2 import MobileNet_v2
from ModelManagement.PytorchModel.Xception import load_Xception
from DataManagement.data_management import DataManagement
from ModelManagement.Util.AverageMeter import *

from ModelManagement.Util.pytorch_util import *
from UtilityManagement.pytorch_utils import *
import os
import time


class ModelManagement:

    def __init__(self):
        self.gpu_check = is_gpu_avaliable()
        print('GPU is available? : ' + str(self.gpu_check))
        self.dev = torch.device("cuda") if self.gpu_check else torch.device("cpu")
        self.start_epoch = 0
        self.total_epoch = 0
        self.current_epoch = 0
        self.ext = None
        self.model = None
        self.state = 'Ready'
        self.summary = ''
        self.image_net_train = None
        self.image_net_validation = None
        self.image_net_test = None
        self.optimizer = None
        self.batch_size = 0
        self.validation_batch_size = 0
        self.learning_rate = 1e-5
        self.train_idx = 0
        self.train_total_idx = 0
        self.train_loss = 0
        self.train_accuracy = 0
        self.train_validation_epoch = 0
        self.train_validation_total_epoch = 0
        self.validation_loss = 0
        self.validation_accuracy = 0
        self.test_batch_size = 0
        self.test_idx_current = 0
        self.test_idx_total = 0
        self.input_shape = 0
        self.test_result = []
        self.validation_result = []
        self.best_prec1 = 0
        self.validation_accuracy_prec5 = 0
        self.train_accuracy_prec5 = 0

        self.train_loader = None
        self.validation_loader = None

        self.criterion = loss_cross_entropy()

        if self.gpu_check:
            self.criterion.to(self.dev)

        pass

    def print_state(self):
        print(self.state)

    def get_test_result(self):
        return self.test_idx_current, self.test_idx_total

    # R-IR-SFR-002
    def get_training_result(self):
        return self.train_idx, self.train_total_idx, self.train_loss, self.train_accuracy, self.train_accuracy_prec5

    def get_epoch(self):
        return self.train_validation_epoch, self.train_validation_total_epoch

    # R-IR-SFR-005
    def get_validation_result(self):
        return self.validation_loss, self.validation_accuracy, self.validation_accuracy_prec5

    def train(self):
        pretrained_path = "./Log"

        if os.path.isfile(os.path.join(pretrained_path, self.model.get_name()+'.pth')):
            print("Pretrained Model Open")
            checkpoint = load_weight_file(os.path.join(pretrained_path, self.model.get_name()+'.pth'))
            self.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            load_weight_parameter(self.model, checkpoint['state_dict'])
            load_weight_parameter(self.optimizer, checkpoint['optimizer'])
        else:
            print("No Pretrained Model")
            self.start_epoch = 0
            self.best_prec1 = 0

        for epoch in range(self.start_epoch, self.total_epoch):
            self.train_validation_epoch = epoch+1
            self.train_validation_total_epoch = self.total_epoch

            # Learning Rate 조절하기
            lr = self.learning_rate * (0.1 ** (epoch // 10))        # ResNet Lerarning Rate
            # lr = self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # train for one epoch
            self.train_per_epoch(self.train_loader, self.model, self.criterion, self.optimizer, epoch, 10)

            # evaluate on validation set
            prec1, prec5 = self.validate(self.validation_loader, self.model, self.criterion, 10)
            self.validation_accuracy = prec1
            self.validation_accuracy_prec5 = prec5

            # remember the best prec@1 and save checkpoint
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.model.get_name(),
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict()
            }, is_best, './Log/'+self.model.get_name(), 'pth')

    def train_per_epoch(self, train_loader, model, criterion, optimizer, epoch, print_freq):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.gpu_check:
                target = target.to(self.dev)
                input = input.to(self.dev)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            self.train_idx = i+1
            self.train_total_idx = int(self.image_net_train.data_num / self.batch_size)
            self.train_loss = loss
            self.train_accuracy = prec1
            self.train_accuracy_prec5 = prec5

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    epoch+1, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

    def validate(self, val_loader, model, criterion, print_freq):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if self.gpu_check:
                target = target.to(self.dev)
                input = input.to(self.dev)
            with torch.no_grad():
                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                self.validation_loss = loss
                self.validation_accuracy = prec1
                self.validation_accuracy_prec5 = prec5

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg, top5.avg

    # R-IR-SFR-007
    def load_model(self, name):
        self.model = None
        if name == 'resnet_18':
            self.model = ResNet18(18, 1000)
        elif name == 'resnet_34':
            self.model = ResNet34(34, 1000)
        elif name == 'resnet_50':
            self.model = ResNet50(50, 1000)
        elif name == 'resnet_101':
            self.model = ResNet101(101, 1000)
        elif name == 'resnet_152':
            self.model = ResNet152(152, 1000)
        elif name == 'vggnet_16':
            self.model = VGG16(16, 1000)
        elif name == 'vggnet_19':
            self.model = VGG19(19, 1000)
        elif name == 'mobilenet_v1':
            self.model = MobileNet_v1(1000, first_channel=32)
        elif name == 'mobilenet_v2':
            self.model = MobileNet_v2(1000)
        elif name == 'xception':
            self.model = load_Xception(1000)
        else:
            self.state = 'PytorchModel is not detected!'
        self.state = 'PytorchModel {} is loaded'.format(name)

        if self.gpu_check:
            print("I use GPU")
            self.model.to(self.dev)

    # R-IR-SFR-008 모델 구성
    def configure_model(self):
        pass

    # R-IR-SFR-009
    def check_model(self):
        if self.gpu_check:
            self.summary = summary(self.model, "cuda")
        else:
            self.summary = summary(self.model, "cpu")
        self.state = 'Pytorch Model Check Finish!'

    def load_test_dataset(self, data_path):
        self.image_net_test = DataManagement(data_path)
        self.state = 'ImageNet Test Dataset Open!'

    def load_validation_dataset(self, data_path):
        self.image_net_validation = DataManagement(data_path)
        self.state = 'ImageNet Validation Dataset Open!'

    def load_train_dataset(self, data_path):
        self.image_net_train = DataManagement(data_path)
        self.state = 'ImageNet Training Dataset Open!'

    # R-IR-SFR-011
    def set_training_parameter(self, learning_rate, epoch, batch_size, num_worker=0):
        self.learning_rate = learning_rate
        self.total_epoch = epoch
        self.batch_size = batch_size
        self.image_net_train.set_batch_size(batch_size)
        self.train_loader = self.image_net_train.get_loader(shuffle=True, num_worker=num_worker)
        self.state = 'Training Setting is Ready!'

    def set_testing_parameter(self, batch_size=1):
        self.test_batch_size = batch_size
        self.image_net_test.set_batch_size(batch_size=batch_size)
        self.image_net_test.get_loader(shuffle=False)
        self.state = 'Testing Setting is Ready!'

    def set_validation_parameter(self, batch_size=1, num_worker=0):
        self.validation_batch_size = batch_size
        self.image_net_validation.set_batch_size(batch_size=batch_size)
        self.validation_loader = self.image_net_validation.get_loader(shuffle=False, num_worker=0)
        self.state = 'Validation Setting is Ready!'

    # R-IR-SFR-010
    def set_optimizer(self, name):
        if name == 'sgd':
            self.optimizer = set_SGD(self.model, self.learning_rate)
        elif name == 'adam':
            self.optimizer = set_Adam(self.model, self.learning_rate)
        elif name == 'adagrad':
            self.optimizer = set_Adagrad(self.model, self.learning_rate)
        elif name == 'rmsprop':
            self.optimizer = set_RMSProp(self.model, self.learning_rate)
        self.state = 'Optimizer {} Open'.format(name)
