import os
import sys
import requests

import numpy as np
import scipy as sp
import scipy.io

import matplotlib.pyplot as plt

import tensorflow as tf

from .datasets import ConditionalDataset

url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
curdir = os.path.abspath(os.path.dirname(__file__))
outdir = os.path.join(curdir, 'files', 'svhn')
outfile = os.path.join(outdir, 'svhn.mat')
testfile = os.path.join(outdir, 'test.mat')
CHUNK_SIZE = 32768

def download_svhn():
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    session = requests.Session()
    response = session.get(url, stream=True)
    print('Downloading: %s' % (url))
    with open(outfile, 'wb') as fp:
        dl = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                dl += len(chunk)
                fp.write(chunk)

                mb = dl / 1.0e6
                sys.stdout.write('\r%.2f MB downloaded...' % (mb))
                sys.stdout.flush()

        sys.stdout.write('\nFinish!\n')
        sys.stdout.flush()

def load_data(datafile = outfile):
    if not os.path.exists(datafile):
        download_svhn()

    mat = sp.io.loadmat(datafile)
    x_train = mat['X']

    x_train = np.transpose(x_train, axes=[3, 0, 1, 2])
    x_train = (x_train / 255.0).astype('float32')

    indices = mat['y']
    indices = np.squeeze(indices)
    indices[indices == 10] = 0
    y_train = np.zeros((len(indices), 10))
    y_train[np.arange(len(indices)), indices] = 1
    y_train = y_train.astype('float32')

    return x_train, y_train

class SVHN(ConditionalDataset):
    def __init__(self):
        super(SVHN, self).__init__()
        self.images, self.attrs = load_data(outfile)
        self.attr_names = [str(i) for i in range(10)]
    def get_test_data(self):
        return load_data(testfile)

