import h5py
import numpy as np

def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]

class Dataset(object):
    def __init__(self):
        self.images = None

    def __len__(self):
        return len(self.images)

    def _get_shape(self):
        return self.images.shape

    def get_test_data(self):
        pass

    shape = property(_get_shape)

class ConditionalDataset(Dataset):
    def __init__(self):
        super(ConditionalDataset, self).__init__()
        self.attrs = None
        self.attr_names = None

class PairwiseDataset(object):
    def __init__(self, x_data, y_data):
        assert x_data.shape[1] == y_data.shape[1]
        assert x_data.shape[2] == y_data.shape[2]
        assert x_data.shape[3] == 1 or y_data.shape[3] == 1 or \
               x_data.shape[3] == y_data.shape[3]

        if x_data.shape[3] != y_data.shape[3]:
            d = max(x_data.shape[3], y_data.shape[3])
            if x_data.shape[3] != d:
                x_data = np.tile(x_data, [1, 1, 1, d])
            if y_data.shape[3] != d:
                y_Data = np.tile(y_data, [1, 1, 1, d])

        x_len = len(x_data)
        y_len = len(y_data)
        l = min(x_len, y_len)

        self.x_data = x_data[:l]
        self.y_data = y_data[:l]

    def __len__(self):
        return len(self.x_data)

    def _get_shape(self):
        return self.x_data.shape

    shape = property(_get_shape)

def load_data(filename, size=-1):
    f = h5py.File(filename)

    dset = ConditionalDataset()
    dset.images = np.asarray(f['images'], 'float32') / 255.0
    dset.attrs = np.asarray(f['attrs'], 'float32')
    dset.attr_names = np.asarray(f['attr_names'])

    if size > 0:
        dset.images = dset.images[:size]
        dset.attrs = dset.attrs[:size]

    return dset
