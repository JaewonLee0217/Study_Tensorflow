import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages
class ResNet:
    def __init__(self, config):
        self._num_residual_units = config.num_residual_units
        self._batch_size = config.batch_size
        self._relu_leakiness = config.relu_leakiness
        self._num_classes = config.num_classes
        self._l2_reg_lambda = config.l2_reg_lambda
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="X")
        self.Y = tf.placeholder(tf.float32, [None, self._num_classes], name="Y")
        self.extra_train_ops = []
    # convonlutional layer 초기화
    # 층별 filter개수(output channel), 최초 convonlutional layer
        filters = [16, 16, 32, 64]
        activate_before_residual = [True, False, False]
        #variable scope는 첫번째 컨볼루션레이어 그룹의 이름을 지어주는 거
        with tf.variable_scope('init'):
            x = self._conv('init_conv', self.X, 3, 3, filters[0], strides=[1, 1, 1, 1])

        #Residual units
        #n=3일 때
        with tf.variable_scope('unit_1_0'):
            x = self._residual(x, filters[0], filters[1], activate_before_residual[0], strides=[1, 1, 1, 1])
        for i in range(1, self._num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = self._residual(x, filters[1], filters[1], strides=[1, 1, 1, 1])

        with tf.variable_scope('unit_2_0'):
            x = self._residual(x, filters[1], filters[2], activate_before_residual[1], strides=[1, 2, 2, 1])
        for i in range(1, self._num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = self._residual(x, filters[2], filters[2], strides=[1, 1, 1, 1])
        with tf.variable_scope('unit_3_0'):
            x = self._residual(x, filters[2], filters[3], activate_before_residual[2], strides=[1, 2, 2, 1])
        for i in range(1, self._num_residual_units):
            with tf.variable_scope('unit_3_%d' % i):
                x = self._residual(x, filters[3], filters[3], strides=[1, 1, 1, 1])
        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, self._relu_leakiness)
            x = self._global_avg_pool(x)

#Fully connected layer, weight decay
        with tf.variable_scope('logit'):
            logits = self._fully_connected(x,self._num_classes)
            #10개의 label 중 하나로 분류하기 위한 Fully connected layer

            self.predictions = tf.nn.softmax(self.predictions,1,name="predictions")
        with tf.variable_scope('loss'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.Y)
            self.loss = tf.reduce_mean(xent,name='xent')
            self.loss += self._decay()

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"),name="accuracy")

    def _residual(self, x, in_filter, out_filter, activate_before_residual=False, strides=[1, 1, 1, 1]):
        if activate_before_residual:
            with tf.variable_scope('common_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self._relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope('residual_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, self._relu_leakiness)
        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, strides)
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, self._relu_leakiness)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                #[1,2,2,1]-> 2x2 kernel, stride =2
                orig_x = tf.nn.avg_pool(orig_x, strides, strides, 'VALID')
                #32채널(현 16채널)
                # 뒤에 16개 채널을 0으로 초기화를 시키자->
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
        x += orig_x
        tf.logging.debug('image after unit %s', x.get_shape())
        return x
#Batch normalization, relu
    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
            self.extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            self.extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _conv(self,name,x,kernel,num_in,num_filters,stride):
        W = tf.get_variable('DW',[kernel,kernel,num_in,num_filters],
                            initializer=tf.initializers.he_normal())
        return con


#Fully connected layer, weight decay
    def _fully_connected(self, x, out_dim):
        dim = tf.reduce_prod(x.get_shape()[1:]).eval()

        x = tf.reshape(x, [-1, dim])
        w = tf.get_variable('DW', [dim, out_dim], initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def _decay(self):
        """L2 weight decay loss."""

        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.multiply(self._l2_reg_lambda, tf.add_n(costs))



