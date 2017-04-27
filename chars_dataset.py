import os, collections, random
from skimage import io
from skimage.filters import threshold_otsu
import numpy as np
from string import ascii_lowercase
import matplotlib
import matplotlib.pyplot as plt


Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def plot_samples(samples):
    """Return a plot of the input samples in a grid.

    The number of samples and sample size must be divisible by 2.

    Parameter
    ---------
    samples : numpy.ndarray
        Must have exactly two dimensions. The first is the sample
        number, while the second is the sample itself (a vector of
        numbers).
    """
    assert samples.shape[0] % 2 == 0,\
        ('Number of samples is not divisible by 2.')
    assert len(samples[0]) % 2 == 0,\
        ('Sample size is not divisible by 2.')

    grid_size = int(np.sqrt(samples.shape[0]))
    img_size = int(np.sqrt(len(samples[0])))

    figure = plt.figure(figsize=(grid_size, grid_size))
    grid = matplotlib.gridspec.GridSpec(grid_size, grid_size)
    grid.update(hspace=0.1, wspace=0.1)

    for idx, sample in enumerate(samples):
        ax = plt.subplot(grid[idx])
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(sample.reshape(img_size, img_size), cmap=plt.cm.gray)

    return figure


class DataSet(object):

    def __init__(self,
                 images,
                 labels):
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def _one_hot(index):
    label = np.zeros(26)
    label[index] = 1
    return label


def _binary_img(img):
    thresh = threshold_otsu(img)
    binary = img < thresh
    binary = binary.astype(np.float32)
    return binary


def _support(projections):
    sup = (projections != 0).astype(np.float32)
    sup = np.add.reduce(sup)
    return sup


def _projection_features(binary):
    row_projections = np.add.reduce(binary, 0)
    col_projections = np.add.reduce(binary, 1)
    r_max = np.max(row_projections)
    r_max_p = np.argmax(row_projections)
    r_mean = np.mean(row_projections)
    r_sup = _support(row_projections)
    c_max = np.max(col_projections)
    c_max_p = np.argmax(col_projections)
    c_mean = np.mean(col_projections)
    c_sup = _support(col_projections)
    return np.array([r_max, r_max_p, r_mean, r_sup, c_max, c_max_p, c_mean, c_sup])


def _transition_features(binary):
    row_gradients = (np.gradient(binary, axis=0) != 0).astype(np.float32)
    col_gradients = (np.gradient(binary, axis=1) != 0).astype(np.float32)
    row_transitions = np.add.reduce(row_gradients, 1)
    col_transitions = np.add.reduce(col_gradients, 1)
    return np.array([np.max(row_transitions), np.max(col_transitions)])


def _rescale(img):
    return np.multiply(img.astype(np.float32), 1.0 / 255.0)


def _extract_data(validation_fraction, test_fraction, data_format ='scaled'):
    data = []
    # Iterate through lowercase alphabet
    for c in ascii_lowercase:
        c_dir = os.getcwd() + '/chars74k-lite/{}'.format(c)
        c_data = []
        for filename in os.listdir(c_dir):
            # Make sure file is a valid data set file
            if filename[:2] != c + '_':
                continue

            img = io.imread(c_dir + '/' + filename)
            if data_format == 'scaled':
                img = _rescale(img)
                c_data.append(img.flatten())
            elif data_format == 'binary':
                binary = _binary_img(img)
                c_data.append(binary.flatten())
            elif data_format == 'compressed':
                binary = _binary_img(img)
                feature_vector = _projection_features(binary)
                feature_vector = np.append(feature_vector, _transition_features(binary))
                c_data.append(feature_vector)

        random.shuffle(c_data)
        data.append(c_data)

    tr = []
    vl = []
    ts = []
    one_hot_index = 0
    for c_data in data:
        l = len(c_data)
        vl_index = int(l*validation_fraction)
        ts_index = int(l - l*test_fraction)
        for img in c_data[:vl_index]:
            vl.append([img, _one_hot(one_hot_index)])
        for img in c_data[vl_index:ts_index]:
            tr.append([img, _one_hot(one_hot_index)])
        for img in c_data[ts_index:]:
            ts.append([img, _one_hot(one_hot_index)])
        one_hot_index += 1
    random.shuffle(tr)
    random.shuffle(vl)
    random.shuffle(ts)

    tr_imgs = np.array([instance[0] for instance in tr])
    tr_lbs = np.array([instance[1] for instance in tr])
    vl_imgs = np.array([instance[0] for instance in vl])
    vl_lbs = np.array([instance[1] for instance in vl])
    ts_imgs = np.array([instance[0] for instance in ts])
    ts_lbs = np.array([instance[1] for instance in ts])

    return tr_imgs, tr_lbs, vl_imgs, vl_lbs, ts_imgs, ts_lbs


def read_data_sets(validation_fraction=0.0, test_fraction=0.2):
    data = _extract_data(validation_fraction, test_fraction)
    train_images, train_labels = data[0], data[1]
    validation_images, validation_labels = data[2], data[3]
    test_images, test_labels = data[4], data[5]
    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return Datasets(train=train, validation=validation, test=test)
