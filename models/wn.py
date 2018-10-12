"""
From pull request:
https://github.com/tensorflow/tensorflow/pull/21276
"""
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.keras.layers import Wrapper
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.eager import context


class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.

    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)

    WeightNorm wrapper works for keras and tf layers.

    ```python
      net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
             input_shape=(32, 32, 3), data_init=True)(x)
      net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                       data_init=True)
      net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                       data_init=True)(net)
      net = WeightNorm(tf.keras.layers.Dense(n_classes),
                       data_init=True)(net)
    ```

    Arguments:
      layer: a `Layer` instance.
      data_init: If `True` use data dependent variable initialization

    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """
    def __init__(self, layer, data_init=True, mean_only_bn=True, training=True, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        self.data_init = data_init
        self.mean_only_bn = mean_only_bn
        self.training = training

        super(WeightNorm, self).__init__(layer, **kwargs)
        self._track_checkpointable(layer, name='layer')

    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.kernel = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def _assign_initialize(self, value):
        if not context.executing_eagerly():
            set_initialized = tf.assign(self.layer.initialized, tf.constant(value))
            return tf.Print(set_initialized, [set_initialized], "assign_initialize for (%s)" % self.layer.g.name)
        else:
            self.layer.initialized = value
        return value

    def _data_dep_init(self, inputs):

        def _do_init():
            if not self.data_init:
                return self._assign_initialize(True)

            """Data dependent initialization for eager execution"""
            from tensorflow.python.ops.nn import moments
            from tensorflow.python.ops.math_ops import sqrt

            with variable_scope.variable_scope('data_dep_init'):
                # Generate data dependent init values
                activation = self.layer.activation
                self.layer.activation = None
                x_init = self.layer.call(inputs)
                m_init, v_init = moments(x_init, self.norm_axes)
                scale_init = 1. / sqrt(v_init + 1e-10)

            # Assign data dependent init values
            self.layer.activation = activation
            assign_ops = [self.layer.g.assign(self.layer.g * scale_init)]
            if self.layer.pop_mean is not None:
                assign_ops.append(self.layer.pop_mean.assign(-m_init * scale_init))
            elif self.layer.bias is not None:
                assign_ops.append(self.layer.bias.assign(-m_init * scale_init))

            with tf.control_dependencies(assign_ops):
                return self._assign_initialize(True)

        return utils.smart_cond(self.layer.initialized, lambda: tf.constant(True), _do_init)

    def _mean_only_batch_norm(self, x, decay=0.9):
        '''
        input comes in which is t=(g*V/||V||)*x
        deterministic : separates training and testing phases
        '''
        with tf.variable_scope('meanOnlyBatchNormalization'):
            moving_mean = self.layer.pop_mean
            b = self.layer.bias
            if not self.training:
                # testing phase, return the result with the accumulated batch mean
                return tf.nn.bias_add(x, b - moving_mean, data_format='NHWC')
            else:
                from tensorflow.python.ops.nn import moments
                # compute the current minibatch mean
                # using convolutional layer as input
                m, _ = moments(x, self.norm_axes)

                # update minibatch mean variable
                moving_mean_op = moving_mean.assign(moving_mean * decay + m * (1 - decay))
                with tf.control_dependencies([moving_mean_op]):
                    return tf.nn.bias_add(x, tf.Print(b - m, [b,m,moving_mean], 'Mean'), data_format='NHWC')

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`WeightNorm` must wrap a layer that'
                    ' contains a `kernel` for weights'
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

            if self.mean_only_bn:
                self.layer.use_bias = False
                self.layer.pop_mean = self.layer.add_variable(
                    name="pop_mean",
                    shape=(self.layer_depth,),
                    initializer=initializers.get('zeros'),
                    use_resource=True,
                    dtype=self.layer.kernel.dtype,
                    trainable=False)
            else:
                self.layer.pop_mean = None

            self.layer.v = self.layer.kernel
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=initializers.get('ones'),
                use_resource=True,
                dtype=self.layer.kernel.dtype,
                trainable=True)


            if context.executing_eagerly():
                self.layer.initialized = False
            else:
                self.layer.initialized = self.layer.add_variable(
                    name="initialized",
                    shape=[],
                    initializer=tf.constant_initializer(False, tf.bool),
                    use_resource=True,
                    dtype=tf.bool,
                    trainable=False
                )

            def assign_init_norm():
                return self.layer.g.assign(self._init_norm(self.layer.v))

            maybe_init_norm = utils.smart_cond(self.layer.initialized, lambda: self.layer.g, assign_init_norm)
            with tf.control_dependencies([maybe_init_norm]):
                self._compute_weights()

            self.layer.built = True

        super(WeightNorm, self).build()
        self.built = True

    def call(self, inputs):
        """Call `Layer`"""

        with ops.control_dependencies([self._data_dep_init(inputs)]):
            if context.executing_eagerly():
                self._compute_weights()  # Recompute weights for each forward pass
            output = self.layer.call(inputs)

        if self.mean_only_bn:
            output = self._mean_only_batch_norm(output)

        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())
