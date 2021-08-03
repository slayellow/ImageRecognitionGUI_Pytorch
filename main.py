from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from mainwindow_ui import Ui_MainWindow
import time

from ModelManagement.model_management import ModelManagement
from UtilityManagement.json import ParsingData
import threading


class MainWindow(Ui_MainWindow):
    def __init__(self, w):
        Ui_MainWindow.__init__(self)

        self.setupUi(w)
        self.qwidget = QWidget()
        self.modelmanagement = ModelManagement()
        self.label_data = ParsingData('data/label_to_content.json')

        self.timer = None
        self.b_train_data = False
        self.b_validation_data = False
        self.b_test_data = False
        self.b_setting = False
        self.b_model = False
        self.b_optimizer = False

        self.BTN_TRAIN.setEnabled(False)
        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(False)
        self.BTN_RESNET34.setEnabled(False)
        self.BTN_RESNET18.setEnabled(False)
        self.BTN_RESNET50.setEnabled(False)
        self.BTN_RESNET101.setEnabled(False)
        self.BTN_RESNET152.setEnabled(False)
        self.BTN_VGG16.setEnabled(False)
        self.BTN_VGG19.setEnabled(False)
        self.BTN_MOBILENET_V1.setEnabled(False)
        self.BTN_MOBILENET_V2.setEnabled(False)
        self.BTN_XCEPTION.setEnabled(False)
        self.TE_MODELCHECK.setFontPointSize(10)

        self.BTN_LOAD_TRAINING_SET.clicked.connect(self.load_training_set)
        self.BTN_LOAD_VALIDATION_SET.clicked.connect(self.load_validation_set)
        self.BTN_RESNET18.clicked.connect(self.load_resnet18)
        self.BTN_RESNET34.clicked.connect(self.load_resnet34)
        self.BTN_RESNET50.clicked.connect(self.load_resnet50)
        self.BTN_RESNET101.clicked.connect(self.load_resnet101)
        self.BTN_RESNET152.clicked.connect(self.load_resnet152)
        self.BTN_VGG16.clicked.connect(self.load_vggnet16)
        self.BTN_VGG19.clicked.connect(self.load_vggnet19)
        self.BTN_XCEPTION.clicked.connect(self.load_xception)
        self.BTN_MOBILENET_V1.clicked.connect(self.load_mobilenet_v1)
        self.BTN_MOBILENET_V2.clicked.connect(self.load_mobilenet_v2)
        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.clicked.connect(self.save_training_parameter)
        self.BTN_OPTIMIZER_SGD.clicked.connect(self.set_optimizer_sgd)
        self.BTN_OPTIMIZER_ADAM.clicked.connect(self.set_optimizer_adam)
        self.BTN_TRAIN.clicked.connect(self.train)
        self.BTN_OPTIMZER_ADAGRAD.clicked.connect(self.set_optimizer_adagrad)
        self.BTN_OPTIMZER_RMSPROP.clicked.connect(self.set_optimizer_rmsprop)

    def set_optimizer_adagrad(self):
        self.modelmanagement.set_optimizer('adagrad')
        self.b_optimizer = True
        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def set_optimizer_rmsprop(self):
        self.modelmanagement.set_optimizer('rmsprop')
        self.b_optimizer = True
        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def timeout(self):
        train_idx, train_total_idx, train_loss, prec1, prec5 = self.modelmanagement.get_training_result()
        epoch, total_epoch = self.modelmanagement.get_epoch()
        validation_loss, validation_accuracy, validation_accuaracy_prec5 = self.modelmanagement.get_validation_result()
        self.LB_TRAINING_EPOCH_CURRENT.setNum(epoch)
        self.LB_TRAINING_EPOCH_TOTAL.setNum(total_epoch)
        self.LB_TRAINING_EPOCH_LOSS.setNum(float(train_loss))
        self.LB_TRAINING_EPOCH_ACCURACY.setNum(float(prec1))
        self.LB_VALIDATION_EPOCH_LOSS.setNum(float(validation_loss))
        self.LB_VALIDATION_EPOCH_ACCURACY.setNum(float(validation_accuracy))
        self.LB_TRAINING_IDX_CURRENT.setNum(train_idx)
        self.LB_TRAINING_IDX_TOTAL.setNum(train_total_idx)
        self.LB_TRAINING_EPOCH_ACCURACY_PREC5.setNum(float(prec5))
        self.LB_VALIDATION_EPOCH_ACCURACY_PREC5.setNum(float(validation_accuaracy_prec5))
        self.timer = threading.Timer(2, self.timeout)
        self.timer.start()

    def on_thread(self):
        self.timer = threading.Timer(2, self.timeout)
        self.timer.start()
        self.modelmanagement.train()
        print("Train Finished!!")
        time.sleep(1)
        self.timer.cancel()

    def train(self):
        t = threading.Thread(target=self.on_thread)
        t.start()

    def set_optimizer_sgd(self):
        self.modelmanagement.set_optimizer('sgd')
        self.b_optimizer = True
        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def set_optimizer_adam(self):
        self.modelmanagement.set_optimizer('adam')
        self.b_optimizer = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

    def save_training_parameter(self):
        learning_rate = float(self.LE_PARAMETER_LEARNING_RATE.text())
        epoch = int(self.LE_PARAMETER_EPOCH.text())
        batch_size = int(self.LE_PARAMETER_BATCH_SIZE.text())

        if self.modelmanagement.image_net_train is not None:
            self.modelmanagement.set_training_parameter(learning_rate=learning_rate, epoch=epoch, batch_size=batch_size
                                                        , num_worker=8)
        if self.modelmanagement.image_net_validation is not None:
            self.modelmanagement.set_validation_parameter(batch_size=batch_size, num_worker=8)
        if self.modelmanagement.image_net_test is not None:
            self.modelmanagement.set_testing_parameter()

    def load_xception(self):
        self.modelmanagement.load_model('xception')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True
        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_vggnet16(self):
        self.modelmanagement.load_model('vggnet_16')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_vggnet19(self):
        self.modelmanagement.load_model('vggnet_19')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_resnet18(self):
        self.modelmanagement.load_model('resnet_18')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_resnet34(self):
        self.modelmanagement.load_model('resnet_34')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_resnet50(self):
        self.modelmanagement.load_model('resnet_50')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_resnet101(self):
        self.modelmanagement.load_model('resnet_101')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_resnet152(self):
        self.modelmanagement.load_model('resnet_152')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_mobilenet_v1(self):
        self.modelmanagement.load_model('mobilenet_v1')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_mobilenet_v2(self):
        self.modelmanagement.load_model('mobilenet_v2')
        self.modelmanagement.check_model()
        self.TE_MODELCHECK.setText(str(self.modelmanagement.summary))
        self.b_model = True

        if self.b_model and self.b_optimizer:
            self.BTN_TRAIN.setEnabled(True)

        self.BTN_TRAINING_PARAMETER_SETTING_SAVE.setEnabled(True)

    def load_validation_set(self):
        data_path = QFileDialog.getExistingDirectory(self.qwidget, "Select Directory")
        self.modelmanagement.load_validation_dataset(data_path)
        self.LB_VALIDATION_DATA_SET_SIZE.setText("Success")

        self.BTN_RESNET34.setEnabled(True)
        self.BTN_RESNET18.setEnabled(True)
        self.BTN_RESNET50.setEnabled(True)
        self.BTN_RESNET101.setEnabled(True)
        self.BTN_RESNET152.setEnabled(True)
        self.BTN_VGG16.setEnabled(True)
        self.BTN_VGG19.setEnabled(True)
        self.BTN_MOBILENET_V1.setEnabled(True)
        self.BTN_MOBILENET_V2.setEnabled(True)

    def load_training_set(self):
        data_path = QFileDialog.getExistingDirectory(self.qwidget, "Select Directory")
        self.modelmanagement.load_train_dataset(data_path)
        self.LB_TRAINING_DATA_SET_SIZE.setText("Success")


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QMainWindow()
    ui = MainWindow(w)
    w.show()
    sys.exit(app.exec_())
