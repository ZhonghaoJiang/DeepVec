import os
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from scipy.io import loadmat
tf.compat.v1.enable_eager_execution()
import numpy as np
from tqdm import tqdm

# augmentation class
class daugor(object):
    def __init__(self, params):
        self.params = params
        self.data_name = params["data_name"]
        (x_train, y_train), (x_test, y_test) = self.load_svhn()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_path = self.params["base_dir"] + "/" + 'x_{}'
        self.y_path = self.params["base_dir"] + "/" + 'y_{}'
        self.op_path = self.params["base_dir"] + "/" + 'op_{}'
        self.gen = None
        self.init_dir()
        self.init_gen()
        # Initialize the folder

    def init_dir(self):
        if not os.path.exists(self.params["base_dir"]):
            os.makedirs(self.params["base_dir"])

    def init_gen(self):
        params_map = {
            "w_r": 0.3,  # Random shift left and right
            "h_r": 0.3,  # Random shift up and down
            "rotation_range": 25,  # Random rotation angle
            "zoom_range": 0.4,  # Random zoom
            "brightness_range": [0.5, 1.5]  # Random brightness change
        }

        gen = ImageDataGenerator(width_shift_range=params_map['w_r'], height_shift_range=params_map['h_r'],
                                 rotation_range=params_map['rotation_range'], zoom_range=params_map['zoom_range'],
                                 fill_mode="constant", brightness_range=params_map["brightness_range"])
        self.gen = gen

    # Get operation name
    def get_op_name(self, op):
        # A. Shift;  B. Center cut;  C. Rotation;  D. Brightness;  E. Contrast;  F. Cut;  G. Flip;  H. Zoom
        op_map = {
            "A": 'Shift',
            "B": 'Center cut',
            "C": 'Rotation',
            "D": 'Adjust Brightness',
            "E": 'Adjust Contrast',
            "F": 'Cut',
            "G": 'Flip',
            "H": 'Zoom',
        }
        return op_map[op]

    # load dataset
    def load_svhn(self):
        train_data = loadmat('../train_32x32.mat')
        test_data = loadmat('../test_32x32.mat')
        x_train = train_data['X'].transpose((3, 0, 1, 2))
        y_train = train_data['y'].flatten() - 1
        x_test = test_data['X'].transpose((3, 0, 1, 2))
        y_test = test_data['y'].flatten() - 1
        return (x_train, y_train), (x_test, y_test)

    # augment data
    def dau_datasets(self, num=10):
        self.init_dir()
        # self.run("train", num=num)
        self.run("test", num=num)

    def run(self, prefix, num=10):
        for i in range(num):
            img_list = []
            label_list = []
            ori_img_list = []
            ori_label_list = []

            if prefix == "train":
                data = zip(self.x_train, self.y_train)
            else:
                data = zip(self.x_test, self.y_test)
            for x, y in tqdm(data):
                img = self.gen.random_transform(x, seed=None)
                img_list.append(img)
                label_list.append(y)
                ori_img_list.append(x)
                ori_label_list.append(y)
            xs = np.array(img_list)
            ys = np.array(label_list)
            xss = np.array(ori_img_list)
            yss = np.array(ori_label_list)
            np.save((self.x_path + "_{}").format(prefix, i), xs)
            np.save((self.y_path + "_{}").format(prefix, i), ys)

            np.save((self.x_path + "_{}").format("ori_test", i), xss)
            np.save((self.y_path + "_{}").format("ori_test", i), yss)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Initialization parameters
    params = {
        "data_name": None,
        "width": None,
        "height": None,
        "channel": None,
        "base_dir": None,
        "model_name": None
    }

    params["data_name"] = "svhn"
    params["width"] = 32
    params["height"] = 32
    params["channel"] = 3
    params["base_dir"] = "dau/{}_harder".format("svhn")
    daugor(params).dau_datasets(num=1)
