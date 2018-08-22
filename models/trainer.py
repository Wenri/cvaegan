import os
import sys
import time
import math
import numpy as np
import tensorflow as tf

from .utils import *

class SemiTrainer(object):
    def __init__(self, datasets, batchsize):
        self.datasets = datasets
        self.current_epoch = tf.Variable(0, name='current_epoch', dtype=tf.int32)
        self.current_batch = tf.Variable(0, name='current_batch', dtype=tf.int32)
        self.batchsize = batchsize
        self.perm = None
        self.test_mode = False

    def init_semi_perm(self, semi_ratio = 0.90):
        self.num_semi = int(len(self.datasets) * semi_ratio)
        self.perm = np.random.permutation(len(self.datasets))
        self.perm_semi = self.perm[:self.num_semi]
        self.perm_full = self.perm[self.num_semi:]
        self.semi_mask = np.zeros(len(self.datasets))
        self.semi_mask[self.perm_semi] = 1
        self.datasets.attrs[self.perm_semi, :] = 0
        return self.perm

    def shuffle_semi_perm(self):
        if self.perm is None:
            return self.init_semi_perm()
        np.random.shuffle(self.perm_semi)
        np.random.shuffle(self.perm_full)
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

    def do_batch(self, model, e, b):
        num_data = len(self.datasets)
        # Check batch size
        bsize = min(self.batchsize, num_data - b)
        indx = self.perm[b:b+bsize]
        if bsize < self.batchsize:
            return

        # Get batch and train on it
        x_batch = model.make_batch(self.datasets, indx)
        losses = model.train_on_batch(x_batch, e * num_data + (b + bsize))
        
        self.print_eta(e, b, bsize)
        self.print_losses(losses)

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
                self.shuffle_semi_perm()
                self.start_time = time.time()
                for b in range(self.current_batch.eval(), num_data, self.batchsize):
                    # Update batch index
                    model.sess.run(update_batch)
                    self.do_batch(model, e, b)
                    if self.test_mode:
                        print('\nFinish testing: %s' % self.name)
                        return      
                print('')
                model.sess.run(update_epoch)

        print('Finished training')
