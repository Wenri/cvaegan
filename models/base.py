import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf

from abc import ABCMeta, abstractmethod
from .utils import *

class BaseModel(metaclass=ABCMeta):
    """
    Base class for non-conditional generative networks
    """

    def __init__(self, **kwargs):
        """
        Initialization
        """
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')
        self.name = kwargs['name']

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.check_input_shape(kwargs['input_shape'])
        self.input_shape = kwargs['input_shape']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.resume = kwargs['resume']

        self.sess = tf.Session()
        self.writer = None
        self.saver = tf.train.Saver()
        self.summary = None

        self.test_size = 10
        self.test_data = None

    def check_input_shape(self, input_shape):
        # Check for CelebA
        if input_shape == (64, 64, 3):
            return

        # Check for MNIST (size modified)
        if input_shape == (32, 32, 1):
            return

        # Check for Cifar10, 100 etc
        if input_shape == (32, 32, 3):
            return

        errmsg = 'Input size should be 32 x 32 or 64 x 64!'
        raise Exception(errmsg)

    def make_batch(self, datasets, indx):
        """
        Get batch from datasets
        """
        return datasets.images[indx]

    def save_images(self, filename):
        """
        Save images generated from random sample numbers
        """
        imgs = self.predict(self.test_data) * 0.5 + 0.5
        imgs = np.clip(imgs, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        _, height, width, dims = imgs.shape

        margin = min(width, height) // 10
        figure = np.ones(((margin + height) * 10 + margin, (margin + width) * 10 + margin, dims), np.float32)

        for i in range(100):
            row = i // 10
            col = i % 10

            y = margin + (margin + height) * row
            x = margin + (margin + width) * col
            figure[y:y+height, x:x+width, :] = imgs[i, :, :, :]

        figure = Image.fromarray((figure * 255.0).astype(np.uint8))
        figure.save(filename)

    def save_model(self, model_file):
        self.saver.save(self.sess, model_file)

    def load_model(self, model_file):
        self.saver.restore(self.sess, model_file)

    @abstractmethod
    def make_test_data(self):
        """
        Please override "make_test_data" method in the derived model!
        """
        pass

    @abstractmethod
    def predict(self, z_sample):
        """
        Please override "predict" method in the derived model!
        """
        pass

    @abstractmethod
    def train_on_batch(self, x_batch, index):
        """
        Please override "train_on_batch" method in the derived model!
        """
        pass

    def image_tiling(self, images, rows, cols):
        n_images = rows * cols
        mg = max(self.input_shape[0], self.input_shape[1]) // 20
        pad_img = tf.pad(images, [[0, 0], [mg, mg], [mg, mg], [0, 0]], constant_values=1.0)
        img_arr = tf.split(pad_img, n_images, 0)

        rows = []
        for i in range(self.test_size):
            rows.append(tf.concat(img_arr[i * cols: (i + 1) * cols], axis=2))

        tile = tf.concat(rows, axis=1)
        return tile

class CondBaseModel(BaseModel):
    def __init__(self, **kwargs):
        super(CondBaseModel, self).__init__(**kwargs)

        if 'attr_names' not in kwargs:
            raise Exception('Please specify attribute names (attr_names')
        self.attr_names = kwargs['attr_names']
        self.num_attrs = len(self.attr_names)

        self.test_size = 10

    def make_batch(self, datasets, indx):
        images = datasets.images[indx]
        attrs = datasets.attrs[indx]
        return images, attrs

    def save_images(self, filename):
        assert self.attr_names is not None

        try:
            test_samples = self.test_data['test_input']
        except KeyError as e:
            print('Key "test_input" must be provided in "make_test_data" method!')
            raise e

        try:
            test_attrs = self.test_data['c_test']
        except KeyError as e:
            print('Key "c_test" must be provided in "make_test_data" method!')
            raise e

        num_test = self.test_size * self.num_attrs
        imgs, _ = self.predict([test_samples[:num_test], test_attrs[:num_test]])
        imgs = np.clip(imgs * 0.5 + 0.5, 0.0, 1.0)

        _, height, width, dims = imgs.shape

        margin = min(width, height) // 10
        figure = np.ones(((margin + height) * self.test_size + margin, (margin + width) * self.num_attrs + margin, dims), np.float32)

        for i in range(self.test_size * self.num_attrs):
            row = i // self.num_attrs
            col = i % self.num_attrs

            y = margin + (margin + height) * row
            x = margin + (margin + width) * col
            figure[y:y+height, x:x+width, :] = imgs[i, :, :, :]

        figure = Image.fromarray((np.squeeze(figure) * 255.0).astype(np.uint8))
        figure.save(filename)
        print('Test Image saved to ' + filename)

    def test_accruacy(self):
        assert self.attr_names is not None

        try:
            test_samples = self.test_data['test_input']
        except KeyError as e:
            print('Key "test_input" must be provided in "make_test_data" method!')
            raise e

        try:
            test_attrs = self.test_data['c_test']
        except KeyError as e:
            print('Key "c_test" must be provided in "make_test_data" method!')
            raise e

        num_test = self.test_size * self.num_attrs
        acc = 0

        for b in range(0, len(test_samples), num_test):
            _, y = self.predict([test_samples[b:b+num_test], test_attrs[b:b+num_test]])
            label_real = np.argmax(test_attrs[b:b+num_test], axis=1)
            label_pred = np.argmax(y, axis=1)
            acc += np.count_nonzero(label_real == label_pred)
            print(label_real)
            print(label_pred)
        
        print("Test accuracy is %d/%d = %.3f" % (acc, len(test_samples), acc/len(test_samples)))
