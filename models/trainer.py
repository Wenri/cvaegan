import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import threading

from sklearn.neighbors import RadiusNeighborsClassifier,KNeighborsClassifier
from sklearn.manifold import TSNE
from .utils import *

def metric_visualization(data_metric, lbls, attr_names, outfile): 
    X_embedded = TSNE(n_components=2).fit_transform(data_metric)
    # plot the result
    vis_x = X_embedded[:, 0]
    vis_y = X_embedded[:, 1]
    n_cls = len(attr_names)
    plt.scatter(vis_x, vis_y, 
        c=lbls, marker='.',
        cmap=plt.cm.get_cmap("jet", n_cls))
    plt.colorbar(ticks=range(n_cls)).set_ticklabels(attr_names)
    plt.clim(-0.5, 9.5)
    plt.savefig(outfile)
    plt.close()

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

class SemiTrainer(object):
    def __init__(self, datasets, batchsize):
        self.datasets = datasets
        self.current_epoch = tf.Variable(0, name='current_epoch', dtype=tf.int32)
        self.current_batch = tf.Variable(0, name='current_batch', dtype=tf.int32)
        self.batchsize = batchsize
        self.perm = None
        self.test_mode = False
        self.data_metric = None
        self.data_labels = None
        self.num_semi = 0
        self.semi_mask = None

    def init_semi_perm(self):
        self.perm = np.random.permutation(len(self.datasets))

        if not hasattr(self.data_labels, 'get_semi_labels'):
            self.semi_mask = None
            self.num_semi = 0
        else:
            semi_tgts, mask_train, mask_confidence = self.datasets.get_semi_labels()
            self.semi_mask = mask_train == 0
            self.num_semi = np.count_nonzero(self.semi_mask)
            self.datasets.attrs[self.semi_mask, :] = 0
            semi_confidence_mask = np.logical_and(self.semi_mask, mask_confidence)
            self.datasets.attrs[semi_confidence_mask, :] = 0.5 * semi_tgts[semi_confidence_mask, :]

        print("Num Semi: %d" % self.num_semi)

        return self.perm

    def shuffle_perm(self):
        if self.perm is None:
            return self.init_semi_perm()
        self.perm = np.random.permutation(len(self.datasets))
        return self.perm

    def prepare_test_data(self, model):
        imgs_t, c_t = self.datasets.get_test_data()
        self.test_data = {'test_input': imgs_t, 'c_test': c_t}
        model.test_data = self.test_data

    def create_directory(self, model):
        # Create output directories if not exist
        self.out_dir = os.path.join(model.output, model.name)
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

        self.res_out_dir = os.path.join(self.out_dir, 'results')
        if not os.path.isdir(self.res_out_dir):
            os.makedirs(self.res_out_dir)

        self.chk_out_dir = os.path.join(self.out_dir, 'checkpoints')
        if not os.path.isdir(self.chk_out_dir):
            os.makedirs(self.chk_out_dir)
        
        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_out_dir = os.path.join(self.out_dir, 'log', time_str)
        if not os.path.isdir(log_out_dir):
            os.makedirs(log_out_dir)

        return log_out_dir

    def print_eta(self, e, b, bsize):
        num_data = len(self.datasets)
        # Print current status
        elapsed_time = time.time() - self.start_time
        eta = elapsed_time / (b + bsize) * (num_data - (b + bsize))
        ratio = 100.0 * (b + bsize) / num_data
        print('Epoch #%d,  Batch: %d / %d (%6.2f %%)  ETA: %s' % \
              (e + 1, b + bsize, num_data, ratio, time_format(eta)))

    def print_losses(self, losses):
        for i, (k, v) in enumerate(losses):
            text = '%s = %8.6f' % (k, v)
            print('  %25s' % (text), end='')
            if (i + 1) % 3 == 0:
                print('')

    def update_metrics(self, model, e):
        num_data = len(self.datasets)
        for b in range(self.current_batch.eval(), num_data, self.batchsize):
            # Check batch size
            bsize = min(self.batchsize, num_data - b)
            x_batch = (self.datasets.images[b:b+bsize], self.datasets.attrs[b:b+bsize])
            ret_batch = model.predict(x_batch)
            self.data_metric[b:b+bsize, :] = ret_batch['z_measure']
            self.data_labels[b:b+bsize, :] = ret_batch['y_predict']

    def metric_visualization(self, e):
        if e % 50 != 0:
            return
        outfile = os.path.join(self.res_out_dir, 'data_metric_epoch_%d' % (e+1))
        np.savez(outfile, data_metric = self.data_metric, data_labels = self.data_labels)

        # threading.Thread(target=metric_visualization, kwargs=dict(
        #     data_metric = self.data_metric[self.perm_full],
        #     lbls = np.argmax(self.datasets.attrs[self.perm_full], axis=1),
        #     attr_names = self.datasets.attr_names,
        #     outfile = os.path.join(self.res_out_dir, 'metric_epoch_%04d.png' % (e + 1))
        #     )
        # ).start()

    def label_propagation(self, model, e):
        if self.num_semi <= 0:
            return
#        radnn = RadiusNeighborsClassifier(radius=2, outlier_label=np.zeros(model.num_attrs))
        radnn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        radnn.fit(self.data_metric[self.semi_mask==0], self.datasets.attrs[self.semi_mask==0])
        lbls = radnn.predict(self.data_metric[self.semi_mask])
        lbls_mask = np.any(lbls, axis=1)
        print("Label Propagation: %d" % np.count_nonzero(lbls_mask))
        data_consider = self.data_labels[self.semi_mask]
        coherence = np.argmax(data_consider, axis=1) == np.argmax(lbls, axis=1)
        coherence[lbls_mask==0] = 0
        confidence = np.max(softmax(data_consider, axis=1), axis=1) > 0.9
        coh_and_cof = np.logical_and(coherence, confidence)
        print("Label Coherence: %d, confidence: %d, coh_and_cof: %d" %
              (np.count_nonzero(coherence), np.count_nonzero(confidence), np.count_nonzero(coh_and_cof)))
        lbls[coh_and_cof == 0] = 0
        self.datasets.attrs[self.semi_mask, :] = 0.5*lbls

    def do_batch(self, model, e, b):
        num_data = len(self.datasets)
        # Check batch size
        bsize = min(self.batchsize, num_data - b)
        indx = self.perm[b:b+bsize]
        if bsize < self.batchsize:
            return

        # Get batch and train on it
        x_batch = model.make_batch(self.datasets, indx)
        ret_batch = model.train_on_batch(x_batch, e * num_data + (b + bsize))
        
        self.print_eta(e, b, bsize)
        self.print_losses(ret_batch.losses)
        # self.update_weights(indx, ret_batch.results)

        print('\n')
        sys.stdout.flush()

        # Save generated images
        save_period = 10000
        if b != 0 and ((b // save_period != (b + bsize) // save_period) or ((b + bsize) == num_data)):
            outfile = os.path.join(self.res_out_dir, 'epoch_%04d_batch_%d.png' % (e + 1, b + bsize))
            model.save_images(outfile)
            model.test_accruacy()
            outfile = os.path.join(self.chk_out_dir, 'epoch_%04d' % (e + 1))
            model.save_model(outfile)

    def main_loop(self, model, epochs=100):
        """
        Main learning loop
        """
        log_out_dir = self.create_directory(model)

        # Make test data
        self.prepare_test_data(model)
        
        # Start training
        with model.sess.as_default():
            model.resume_model()

            # Update rule
            num_data = len(self.datasets)
            update_epoch = self.current_epoch.assign(self.current_epoch + 1)
            update_batch = self.current_batch.assign(tf.mod(tf.minimum(self.current_batch + self.batchsize, num_data), num_data))

            model.writer = tf.summary.FileWriter(log_out_dir, model.sess.graph)
            model.sess.graph.finalize()

            print('\n\n--- START TRAINING ---\n')
            for e in range(self.current_epoch.eval(), epochs):
                self.data_metric = np.zeros((len(self.datasets), model.metric_dims), dtype=np.float32)
                self.data_labels = np.zeros((len(self.datasets), model.num_attrs), dtype=np.float32)
                self.shuffle_perm()
                self.start_time = time.time()
                for b in range(self.current_batch.eval(), num_data, self.batchsize):
                    # Update batch index
                    model.sess.run(update_batch)
                    self.do_batch(model, e, b)
                    if self.test_mode:
                        print('\nFinish testing: %s' % self.name)
                        return
                print('')
                if e+1 >= 50:
                    self.update_metrics(model, e)
                    self.metric_visualization(e)
                    self.label_propagation(model, e)
                model.sess.run(update_epoch)

        print('Finished training')
