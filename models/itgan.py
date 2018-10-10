import math
import numpy as np
import tensorflow as tf

from types import SimpleNamespace
from .base import CondBaseModel
from .utils import *
from .nn import conv2d
#from ..appprop import appProp

class Encoder(object):
    def __init__(self, input_shape, z_dims, metric_dims, num_attrs):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.z_dims = z_dims
        self.metric_dims = metric_dims
        self.num_attrs = num_attrs
        self.name = 'encoder'


    def _conv(self, inputs, filter, name = None, w = 5, s = 1, training=True, padding='same'):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs, filter, (w, w), (s, s), padding)
            x = tf.layers.batch_normalization(x, training=training)
            x = lrelu(x, 0.1)
        return x

    def _convwn(self, inputs, filter, name = None, w = 5, s = 1, training=True, padding='SAME'):
        x = conv2d(inputs, filter, [w, w], stride=[s, s], pad=padding, nonlinearity=lrelu,
                   use_weight_normalization=True, use_mean_only_batch_normalization=True, name=name)
        return x

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            x = gaussian_noise_layer(inputs, 0.15)
            x = self._conv(x, 128, 'conv1a', 3, 1, training)
            x = self._convwn(x, 128, 'conv1b', 3, 1, training)
            x = self._convwn(x, 128, 'conv1c', 3, 1, training) # 32 x 32
            x = tf.layers.max_pooling2d(x, 2, 2, name='pool1')
            x = tf.layers.dropout(x, name='drop1', training=training)

            x = self._convwn(x, 256, 'conv2a', 3, 1, training)
            x = self._convwn(x, 256, 'conv2b', 3, 1, training)
            x = self._convwn(x, 256, 'conv2c', 3, 1, training) # 16 x 16
            x = tf.layers.max_pooling2d(x, 2, 2, name='pool2')
            x = tf.layers.dropout(x, name='drop2', training=training)

            x = self._convwn(x, 512, 'conv3a', 3, 1, training, 'VALID')
            x = self._convwn(x, 256, 'conv3b', 1, 1, training)
            x = self._convwn(x, self.metric_dims, 'conv3c', 1, 1, training) # 8 x 8

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                y = tf.layers.dense(x, self.num_attrs, name='y_predict')
                z_avg = tf.layers.dense(x, self.z_dims, name='z_avg')
                z_log_var = tf.layers.dense(x, self.z_dims, name='z_log_var')

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True

        return z_avg, z_log_var, y, x

class Decoder(object):
    def __init__(self, input_shape):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.name = 'decoder'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('fc1'):
                w = self.input_shape[0] // (2 ** 3)
                x = tf.layers.dense(inputs, w * w * 256)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)
                x = tf.reshape(x, [-1, w, w, 256])

            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d_transpose(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d_transpose(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d_transpose(x, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = lrelu(x)

            with tf.variable_scope('conv4'):
                d = self.input_shape[2]
                x = tf.layers.conv2d_transpose(x, d, (5, 5), (1, 1), 'same')
                x = tf.tanh(x)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True

        return x

class Classifier(object):
    def __init__(self, input_shape, num_attrs):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.name = 'classifier'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                f = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(f, self.num_attrs)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True

        return y, f

class Discriminator(object):
    def __init__(self, input_shape):
        self.variables = None
        self.reuse = False
        self.input_shape = input_shape
        self.name = 'discriminator'

    def __call__(self, inputs, training=True):
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('conv1'):
                x = tf.layers.conv2d(inputs, 64, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv2'):
                x = tf.layers.conv2d(x, 128, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv3'):
                x = tf.layers.conv2d(x, 256, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('conv4'):
                x = tf.layers.conv2d(x, 512, (5, 5), (2, 2), 'same')
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.relu(x)

            with tf.variable_scope('global_avg'):
                x = tf.reduce_mean(x, axis=[1, 2])

            with tf.variable_scope('fc1'):
                f = tf.contrib.layers.flatten(x)
                y = tf.layers.dense(f, 1)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
        self.reuse = True

        return y, f


class iTGAN(CondBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 128,
        name='cvaegan',
        trainer = None,
        **kwargs
    ):
        super(iTGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.trainer = trainer

        self.z_dims = z_dims
        self.metric_dims = z_dims

        self.alpha = 0.7

        self.E_f_D_r = None
        self.E_f_D_p = None
        self.E_f_C_r = None
        self.E_f_C_p = None

        self.f_enc = None
        self.f_gen = None
        self.f_cls = None
        self.f_dis = None

        self.x_r = None
        self.c_r = None
        self.z_p = None

        self.z_test = None
        self.x_test = None
        self.c_test = None

        self.enc_trainer = None
        self.gen_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.gen_loss = None
        self.dis_loss = None
        self.gen_acc = None
        self.dis_acc = None

        self.build_model()

    def train_on_batch(self, batch, index):
        x_r, c_r = batch
        batchsize = len(x_r)
        # z_p = np.random.uniform(-1, 1, size=(len(x_r), self.z_dims))
        # _, _, dis_loss, dis_acc, z_measure = self.sess.run(
        #     (self.dis_trainer, self.cls_trainer, self.dis_loss, self.dis_acc, self.z_measure),
        #     feed_dict={
        #         self.x_r: x_r, self.c_r: c_r
        #     }
        # )
        # _, _, gen_loss, gen_acc, z_measure = self.sess.run(
        #     (self.gen_trainer, self.enc_trainer, self.gen_loss, self.gen_acc, self.z_measure),
        #     feed_dict={
        #         self.x_r: x_r, self.c_r: c_r
        #     }
        # )
        _, gen_loss, z_measure = self.sess.run(
            (self.enc_trainer, self.gen_loss, self.z_measure),
            feed_dict={
                self.x_r: x_r, self.c_r: c_r
            }
        )
        gen_acc = 0
        dis_loss = 0
        dis_acc = 0
        summary_priod = 1000
        if index // summary_priod != (index + batchsize) // summary_priod:
            test_samples = self.test_data['test_input']
            test_attrs = self.test_data['c_test']
            num_test = self.test_size * self.num_attrs
            summary = self.sess.run(
                self.summary,
                feed_dict={
                    self.x_r: x_r, self.c_r: c_r,
                    self.test_input: test_samples[:num_test], self.c_test: test_attrs[:num_test]
                }
            )
            self.writer.add_summary(summary, index)

        return SimpleNamespace(
            losses = [
                ('gen_loss', gen_loss), ('dis_loss', dis_loss),
                ('gen_acc', gen_acc), ('dis_acc', dis_acc)
            ],
            results = {
                'z_measure': z_measure
            }
        )

    def predict(self, batch):
        z_samples, c_samples = batch
        fetches = {
            'x_test': self.x_test,
            'y_predict': self.c_test_pred,
            'z_measure': self.z_test_measure
        }
        return self.sess.run(
            fetches, feed_dict={
                self.test_input: z_samples,
                self.c_test: c_samples
            }
        )

    def make_test_data(self):
        c_t = np.identity(self.num_attrs)
        c_t = np.tile(c_t, (self.test_size, 1))
        z_t = np.random.normal(size=(self.test_size, self.z_dims))
        z_t = np.tile(z_t, (1, self.num_attrs))
        z_t = z_t.reshape((self.test_size * self.num_attrs, self.z_dims))
        self.test_data = {'z_test': z_t, 'c_test': c_t}

    def build_model(self):
        self.f_enc = Encoder(self.input_shape, self.z_dims, self.metric_dims, self.num_attrs)
        self.f_gen = Decoder(self.input_shape)

        self.f_cls = Classifier(self.input_shape, self.num_attrs)
        self.f_dis = Discriminator(self.input_shape)

        # Trainer
        self.x_r = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        self.c_r = tf.placeholder(tf.float32, shape=(None, self.num_attrs))

        z_avg, z_log_var, self.y_pred, self.z_measure = self.f_enc(self.x_r)

        z_f = sample_normal(z_avg, z_log_var)
        x_f = self.f_gen(z_f)

        #self.z_p = tf.placeholder(tf.float32, shape=(None, self.z_dims))
        #x_p = self.f_gen(self.z_p, self.c_r)

        c_r_pred, f_C_r = self.f_cls(self.x_r)
        c_f, f_C_f = self.f_cls(x_f)
        #c_p, f_C_p = self.f_cls(x_p)

        y_r, f_D_r = self.f_dis(self.x_r)
        y_f, f_D_f = self.f_dis(x_f)
        #y_p, f_D_p = self.f_dis(x_p)

        L_KL = kl_loss(z_avg, z_log_var)

        y_predone = tf.one_hot(tf.argmax(self.y_pred, axis=1), self.num_attrs)
        c_weights = tf.reduce_max(self.c_r, axis=1)
        c_semi = tf.where(self.c_r > 0.01, tf.ones_like(self.c_r), tf.zeros_like(self.c_r))

        enc_opt = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.9)
        gen_opt = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.9)
        cls_opt = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.9)
        dis_opt = tf.train.AdamOptimizer(learning_rate=1.0e-3, beta1=0.9)

        #c_weights = tf.maximum(c_weights, 0.01)
        #c_weights = tf.maximum(c_weights, tf.minimum(tf.cast(self.trainer.current_epoch, tf.float32) / 100, 0.8))
        
        L_GD = self.L_GD(f_D_r, f_D_f)
        L_GC = self.L_GC(f_C_r, f_C_f, self.c_r)
        L_G = self.L_G(self.x_r, x_f, f_D_r, f_D_f, f_C_r, f_C_f)
        L_GT = self.L_Metric(self.z_measure, c_semi, c_weights > 0.1)

        with tf.name_scope('L_rec'):
            # L_rec =  0.5 * tf.losses.mean_squared_error(self.x_r, x_f)
            L_rec = 0.1 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.x_r, x_f), axis=[1, 2, 3]))

        with tf.name_scope('L_D'):
            L_D = tf.losses.sigmoid_cross_entropy(tf.ones_like(y_r), y_r) + \
                  tf.losses.sigmoid_cross_entropy(tf.zeros_like(y_f), y_f)

        with tf.name_scope('L_C'):
            L_C = tf.losses.softmax_cross_entropy(c_semi, c_r_pred, weights=c_weights)

        with tf.name_scope('L_CPC'):
            L_CPC = tf.losses.softmax_cross_entropy(c_semi, self.y_pred, weights=c_weights)

        with tf.control_dependencies(self.f_enc.update_ops):
            # self.enc_trainer = enc_opt.minimize(L_G + L_KL + L_GT + L_CPC, var_list=self.f_enc.variables)
            # self.enc_trainer = enc_opt.minimize(L_rec + L_KL + L_GT + L_CPC, var_list=self.f_enc.variables)
            self.enc_trainer = enc_opt.minimize(L_GT + L_CPC, var_list=self.f_enc.variables)
        with tf.control_dependencies(self.f_gen.update_ops):
            # self.gen_trainer = gen_opt.minimize(L_G + L_GD + L_GC, var_list=self.f_gen.variables)
            self.gen_trainer = gen_opt.minimize(L_rec, var_list=self.f_gen.variables)
        with tf.control_dependencies(self.f_cls.update_ops):
            self.cls_trainer = cls_opt.minimize(L_C, var_list=self.f_cls.variables)
        with tf.control_dependencies(self.f_dis.update_ops):
            self.dis_trainer = dis_opt.minimize(L_D, var_list=self.f_dis.variables)

        # self.gen_loss = L_G + L_GD + L_GC + L_GT + L_CPC
        self.gen_loss = L_GT + L_CPC
        self.dis_loss = L_D

        # Predictor
        self.test_input = tf.placeholder(tf.float32, shape=(None,) + self.input_shape)
        self.c_test = tf.placeholder(tf.float32, shape=(None, self.num_attrs))
        z_test_avg, z_test_log_var, self.c_test_pred, self.z_test_measure = self.f_enc(self.test_input)
        z_test = sample_normal(z_test_avg, z_test_log_var)
        self.x_test = self.f_gen(z_test)
        x_tile = self.image_tiling(self.x_test, self.test_size, self.num_attrs)

        # Summary
        tf.summary.image('x_real', self.x_r, 10)
        tf.summary.image('x_fake', x_f, 10)
        tf.summary.image('x_tile', x_tile, 1)
        tf.summary.scalar('L_G', L_G)
        tf.summary.scalar('L_GD', L_GD)
        tf.summary.scalar('L_GC', L_GC)
        tf.summary.scalar('L_C', L_C)
        tf.summary.scalar('L_D', L_D)
        tf.summary.scalar('L_KL', L_KL)
        tf.summary.scalar('gen_loss', self.gen_loss)
        tf.summary.scalar('dis_loss', self.dis_loss)

        # Accuracy
        self.gen_acc = 0.5 * binary_accuracy(tf.ones_like(y_f), y_f)

        self.dis_acc = binary_accuracy(tf.ones_like(y_r), y_r) / 2.0 + \
                       binary_accuracy(tf.zeros_like(y_f), y_f) / 2.0

        tf.summary.scalar('gen_acc', self.gen_acc)
        tf.summary.scalar('dis_acc', self.dis_acc)

        self.summary = tf.summary.merge_all()

    def L_G(self, x_r, x_f, f_D_r, f_D_f, f_C_r, f_C_f):
        with tf.name_scope('L_G'):
            loss = tf.constant(0.0, dtype=tf.float32)
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x_r, x_f), axis=[1, 2, 3]))
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_D_r, f_D_f), axis=[1]))
            loss += 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(f_C_r, f_C_f), axis=[1]))

        return loss

    def L_GD(self, f_D_r, f_D_p):
        with tf.name_scope('L_GD'):
            # Compute loss
            E_f_D_r = tf.reduce_mean(f_D_r, axis=0)
            E_f_D_p = tf.reduce_mean(f_D_p, axis=0)

            # Update features
            if self.E_f_D_r is None:
                self.E_f_D_r = tf.zeros_like(E_f_D_r)

            if self.E_f_D_p is None:
                self.E_f_D_p = tf.zeros_like(E_f_D_p)

            self.E_f_D_r = self.alpha * self.E_f_D_r + (1.0 - self.alpha) * E_f_D_r
            self.E_f_D_p = self.alpha * self.E_f_D_p + (1.0 - self.alpha) * E_f_D_p
            return 0.5 * tf.reduce_sum(tf.squared_difference(self.E_f_D_r, self.E_f_D_p))

    def L_GC(self, f_C_r, f_C_p, c):
        with tf.name_scope('L_GC'):
            image_shape = tf.shape(f_C_r)
            print('image_shape: ', f_C_r.shape)

            indices = tf.eye(self.num_attrs, dtype=tf.float32)
            indices = tf.tile(indices, (1, image_shape[0]))
            indices = tf.reshape(indices, (-1, self.num_attrs))
            print('indices: ', indices.shape)

            classes = tf.tile(c, (self.num_attrs, 1))
            print('classes: ', classes.shape)

            mask = tf.reduce_max(tf.multiply(indices, classes), axis=1)
            mask = tf.reshape(mask, (-1, 1))
            mask = tf.tile(mask, (1, image_shape[1]))
            print('mask: ', mask.shape)

            denom = tf.reshape(tf.multiply(indices, classes), (self.num_attrs, image_shape[0], self.num_attrs))
            denom = tf.reduce_sum(denom, axis=[1, 2])
            denom = tf.tile(tf.reshape(denom, (-1, 1)), (1, image_shape[1]))
            print('denom: ', denom.shape)

            f_1_sum = tf.tile(f_C_r, (self.num_attrs, 1)) # b*n,f
            f_1_sum = tf.multiply(f_1_sum, mask) # b*n,f x b*n,f
            f_1_sum = tf.reshape(f_1_sum, (self.num_attrs, image_shape[0], image_shape[1]))
            print('f_1_sum: ', f_1_sum.shape)
            E_f_1 = tf.divide(tf.reduce_sum(f_1_sum, axis=1), denom + 1.0e-8)
            print('E_f_1: ', E_f_1.shape)

            f_2_sum = tf.tile(f_C_p, (self.num_attrs, 1))
            f_2_sum = tf.multiply(f_2_sum, mask)
            f_2_sum = tf.reshape(f_2_sum, (self.num_attrs, image_shape[0], image_shape[1]))
            E_f_2 = tf.divide(tf.reduce_sum(f_2_sum, axis=1), denom + 1.0e-8)

            # Update features
            if self.E_f_C_r is None:
                self.E_f_C_r = tf.zeros_like(E_f_1)

            if self.E_f_C_p is None:
                self.E_f_C_p = tf.zeros_like(E_f_2)

            self.E_f_C_r = self.alpha * self.E_f_C_r + (1.0 - self.alpha) * E_f_1
            self.E_f_C_p = self.alpha * self.E_f_C_p + (1.0 - self.alpha) * E_f_2

            # return 0.5 * tf.losses.mean_squared_error(self.E_f_C_r, self.E_f_C_p)
            return 0.5 * tf.reduce_sum(tf.squared_difference(self.E_f_C_r, self.E_f_C_p))

    def L_Metric(self, z_measure, c, mask):
        with tf.name_scope('L_GT'):
            c_m = tf.boolean_mask(tf.Print(c, [tf.argmax(c, axis=1), mask], "SemiLabel"), mask)
            z_m = tf.boolean_mask(self.z_measure, mask)
            n_labels = tf.count_nonzero(tf.reduce_sum(c_m, axis=0) > 0.1)
            two_labels = tf.count_nonzero(tf.reduce_sum(c_m, axis=0) > 1.1)
            cond = tf.logical_and(two_labels>0, n_labels>1)
            return tf.cond(tf.Print(cond, [n_labels, two_labels, cond], 'Labels: '),
                lambda: tf.reduce_mean(
                    tf.reduce_max(c_m, axis=1)
                ) * tf.contrib.losses.metric_learning.triplet_semihard_loss(
                    labels = tf.argmax(c_m, axis=1), embeddings = z_m, margin=0.2
                ),
                lambda: tf.constant(0, dtype=tf.float32))
