import tensorflow as tf
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

# GPU 세팅
def set_gpu_setting():
    if not tf.test.gpu_device_name():
        print("No GPU Found!")
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)


# R-IR-SFR-004
def visualization(image_path, prediction, prediction_label=None):
    if prediction_label is None:
        image_path = [tf.compat.as_str_any(tensor.numpy()) for tensor in image_path]
        image = img.imread(image_path[0])
        plt.imshow(image)
        plt.title('Path : {}'.format(image_path), fontsize=8)
        plt.xlabel("Prediction : {}".format(int(prediction)))
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        image_path = [tf.compat.as_str_any(tensor.numpy()) for tensor in image_path]
        image = img.imread(image_path[0])
        plt.imshow(image)
        plt.title('Path : {}'.format(image_path), fontsize=8)
        plt.xlabel("Prediction : {}".format(int(prediction)))
        plt.ylabel("Label : {}".format(int(prediction_label)))
        plt.show(block=False)
        plt.pause(1)
        plt.close()
