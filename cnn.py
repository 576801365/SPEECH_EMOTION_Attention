import tensorflow as tf
from tensorflow.python.training import moving_averages

#IO
#===============================================================
tf.app.flags.DEFINE_string('datapath', '.IEMOCAP_Mel_delta12.pkl', 'path of mel-frames')
tf.app.flags.DEFINE_string('savepath', './model/', '')
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', '')
tf.app.flags.DEFINE_string('pred_name', './pred0.pkl', '')

#Global Constants
#================================================================
#dropout
tf.app.flags.DEFINE_float('train_dropout', 0.7, '')
tf.app.flags.DEFINE_float('test_dropout', 1.0, '')
tf.app.flags.DEFINE_float('dropout_conv', 1, '')
tf.app.flags.DEFINE_float('dropout_full1', 0.7, '')
tf.app.flags.DEFINE_float('dropout_full2', 0.7, '')
tf.app.flags.DEFINE_float('dropout_lstm', 0.7, '')
#decay learning rate
tf.app.flags.DEFINE_float('learning_rate_decay_rate', 0.99, '')
tf.app.flags.DEFINE_integer('learning_rate_decay_step', 500, '')
tf.app.flags.DEFINE_float('epsilon', 1e-8, 'epsilon parameter of Adam optimizer')
#train constants
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('epoch_num', 20000, '')
#batch size
tf.app.flags.DEFINE_integer('train_batch_size', 17, '')
tf.app.flags.DEFINE_integer('test_batch_size', 40, '')
#data shape
tf.app.flags.DEFINE_integer('inputdata_channal', 3, '')
tf.app.flags.DEFINE_integer('inputdata_length', 300, '')
tf.app.flags.DEFINE_integer('inputdata_weight', 40, '')
#network shape
tf.app.flags.DEFINE_integer('linear_num', 786, 'hidden number of linear layer')
tf.app.flags.DEFINE_integer('seq_len', 150, 'sequence length of lstm')
tf.app.flags.DEFINE_integer('cell_num', 128, 'cell units of the lstm')
tf.app.flags.DEFINE_integer('fully1_shape', 64, '')
tf.app.flags.DEFINE_integer('fully2_shape', 4, '')
tf.app.flags.DEFINE_integer('attention_size', 1, '')

FLAGS = tf.app.flags.FLAGS


class CRNN(object):
    def __init__(self, model):
        self.model = model
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.inputdata_length, FLAGS.inputdata_weight, FLAGS.inputdata_channal])
        self.inputs_label = tf.placeholder(tf.int8, [None, 4])
        self.keep_prob = tf.placeholder(tf.float32)
        self._extra_train_ops = []

    def _conv_2d(self, name, x, filter_size, in_channal_num, out_channal_num, strides):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weight',
                                     shape=[filter_size[0], filter_size[1], in_channal_num, out_channal_num],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0.0, stddev=0.1))
            bias = tf.get_variable(name='bias',
                                   shape=[out_channal_num],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            conv_2 = tf.nn.conv2d(x, weight, [1, strides[0], strides[1], 1], padding='SAME')
            return tf.nn.bias_add(conv_2, bias)

    def _max_pool(self, x, pool_size, strides):
        return tf.nn.max_pool(x, ksize=[1, pool_size[0], pool_size[1], 1],
                              strides=[1, strides[0], strides[1], 1],
                              padding='VALID',
                              name='max_pool')

    def _linear(self, x, name, shape):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weight',
                                     shape=shape,
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0.0, stddev=0.1))
            bias = tf.get_variable(name='bias',
                                   shape=shape[1],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0, dtype=tf.float32))
            return tf.matmul(tf.nn.dropout(x, self.keep_prob), weight) + bias

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _batch_norm(self, name, x):    #CNN batch normalization
        with tf.variable_scope(name):
            num_channals = [x.get_shape()[-1]]
            gamma = tf.get_variable('gamma', num_channals, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            beta = tf.get_variable('beta', num_channals, tf.float32,
                                    initializer=tf.constant_initializer(0.0, tf.float32))
            if (self.model=='train'):
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                movingmean_mean = tf.get_variable('movingmean_mean', num_channals, tf.float32,
                                                  initializer=tf.constant_initializer(0.0, tf.float32),
                                                  trainable=False)
                movingmean_variance = tf.get_variable('movingmean_variance', num_channals, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                    trainable=False)
#                self._extra_train_ops.append(moving_averages.assign_moving_average(mean, movingmean_mean, 0.9))
#                self._extra_train_ops.append(moving_averages.assign_moving_average(variance, movingmean_variance, 0.9))
            else:#test sample without batch normalization
                mean = tf.get_variable('movingmean_mean', num_channals, tf.float32,
                                                  initializer=tf.constant_initializer(0.0, tf.float32),
                                                  trainable=False)
                variance = tf.get_variable('movingmean_variance', num_channals, tf.float32,
                                                  initializer=tf.constant_initializer(1.0, tf.float32),
                                                  trainable=False)
            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            x_bn.set_shape(x.get_shape())
            return x_bn

    def _batch_norm_linear(self, name, x, decay=0.999):
        with tf.variable_scope(name):
            gamma = tf.Variable(tf.ones([x.get_shape()[-1]]))
            beta = tf.Variable(tf.zeros([x.get_shape()[-1]]))
            variance = tf.Variable(tf.ones([x.get_shape()[-1]]), trainable=False)
            mean = tf.Variable(tf.zeros([x.get_shape()[-1]]), trainable=False)

            if (self.model=='train'):
                batch_mean, batch_variance = tf.nn.moments(x, [0])
                train_mean = tf.assign(mean, mean * decay + batch_mean * (1-decay))
                train_variance = tf.assign(variance, variance * decay + batch_mean * (1 - decay))
                with tf.control_dependencies([train_mean, train_variance]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_variance, beta, gamma, FLAGS.epsilon)
            else:
                return tf.nn.batch_normalization(x, mean, variance, beta, gamma, FLAGS.epsilon)

    def _attention(self, inputs, attention_size):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        hidden_size = inputs.shape[2].value

        W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1)
        alphas = tf.nn.softmax(vu)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        return output

    def _build_model(self):
        filter_channalnum = [128, 512]
        filter_size = [5, 3]
        filter_stride = [1, 1]
        pool1_size = [2, 4]
        pool2_size = [1, 2]
        p = 5 #max_pool2output.shape [batrch_num, 150, 5, 1]
        with tf.variable_scope('cnn'):
            with tf.variable_scope('unit_1'):
                x1 = self._conv_2d('cnn1', self.inputs,
                                  filter_size, FLAGS.inputdata_channal, filter_channalnum[0],
                                  filter_stride)
                x1_bn   = self._batch_norm('bn1', x1)
                x1_relu = self._leaky_relu(x1_bn, 0.01)
                x1_pool = self._max_pool(x1_relu, pool1_size, pool1_size)
#               print(x1_pool.get_shape())
            with tf.variable_scope('unit_2'):
                x2 = self._conv_2d('cnn1', x1_pool,
                                  filter_size, filter_channalnum[0], filter_channalnum[1],
                                  filter_stride)
                x2_bn   = self._batch_norm('bn2', x2)
                x2_relu = self._leaky_relu(x2_bn, 0.01)
                x2_pool = self._max_pool(x2_relu, pool2_size, pool2_size)

            with tf.variable_scope('linear'):
                xl_in = tf.reshape(x2_pool, [-1, p*filter_channalnum[1]])
                xl = self._linear(xl_in, 'linear1', [p*filter_channalnum[1], FLAGS.linear_num])

            with tf.variable_scope('lstm'):
                xrnn_in = tf.reshape(xl, [-1, FLAGS.seq_len, FLAGS.linear_num])
                cell_fw = tf.contrib.rnn.BasicLSTMCell(FLAGS.cell_num, forget_bias=1.0)
                #if self.model == 'train':
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell=cell_fw, output_keep_prob=self.keep_prob)
                cell_bw = tf.contrib.rnn.BasicLSTMCell(FLAGS.cell_num, forget_bias=1.0)
                #if self.model == 'train':
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell=cell_bw, output_keep_prob=self.keep_prob)

                output, output_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                       cell_bw,
                                                                       xrnn_in,
                                                                       dtype=tf.float32,
                                                                       time_major=False,
                                                                       scope='lstm1')
                with tf.variable_scope('time_pool'):
                    #output = tf.concat(output, 2)
                    output = self._attention(output, FLAGS.attention_size)
                    #output = tf.reshape(output, [-1, FLAGS.seq_len, 2*FLAGS.cell_num, 1])
                    #output = self._max_pool(output, [FLAGS.seq_len, 1], [FLAGS.seq_len, 1])
                    #output = tf.reshape(output, [-1, 2*FLAGS.cell_num])

                with tf.variable_scope('dense'):
                    y1 = self._linear(output, 'dense_matul', [2*FLAGS.cell_num, FLAGS.fully1_shape])
                    y1 = self._batch_norm_linear('dense_bn', y1)
                    y1 = self._leaky_relu(y1, 0.01)

                self.logits = self._linear(y1, 'softmax', [FLAGS.fully1_shape, FLAGS.fully2_shape])
