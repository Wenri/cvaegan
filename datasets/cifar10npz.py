
import numpy as np
import os
from .datasets import ConditionalDataset, one_hot_encoded
from .cifar10 import load_class_names

########################################################################
num_classes = 10

class load_data(ConditionalDataset):
    def __init__(self, npz_file='prepare_dataset.npz'):
        super(load_data, self).__init__()
        self.d = np.load(npz_file)
        self.images = self.d['X_train'].transpose([0, 2, 3, 1])
        self.attrs = one_hot_encoded(class_numbers=self.d['y_train'],
                                     num_classes=num_classes)
        self.attr_names = load_class_names()

    def get_test_data(self):
        images = self.d['X_test'].transpose([0, 2, 3, 1])
        attrs = one_hot_encoded(class_numbers=self.d['y_test'],
                                num_classes=num_classes)
        return images, attrs

    def get_semi_labels(self, em_file='ensemble_prediction_epoch_299.npz'):
        em = np.load(em_file)
        pred = em['ensemble_prediction']
        tgts = one_hot_encoded(class_numbers=np.argmax(pred, axis=1),
                               num_classes=num_classes)
        masm = np.max(pred, axis=1) > 0.9
        return tgts, self.d['mask_train'], masm
