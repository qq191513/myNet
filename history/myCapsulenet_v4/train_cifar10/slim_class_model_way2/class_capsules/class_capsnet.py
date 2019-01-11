from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from class_capsules.class_capsules_layer import capsule_layer

from tensorflow.contrib.layers.python.layers import initializers




class capsules_v0(object):
    def __init__(self,images,labels,logdir = './log',NUM_TRAIN_EXAMPLES=50000,
                 batch_size=24,is_training=True):
        self.images = tf.reshape(images,[-1,32,32,3])
        self.labels = labels
        self.NUM_TRAIN_EXAMPLES = NUM_TRAIN_EXAMPLES
        self.batch_size = batch_size
        self.summary_save_path = logdir
        n_classes = 10

        if is_training:
            # self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.global_step = tf.train.get_or_create_global_step()
            # self.images = tf.placeholder(tf.float32, shape=[self.batch_size, 32 , 32 ,3], name='Input')
            # self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, n_classes], name='labels')
            self.capsule_layer = capsule_layer(self)
            self.bulld_arch(num_classes=n_classes, iterations=1, name='capsulesEM-V0')
            self.spread_loss(name='spread_loss')
            self.get_accuracy()
            self._summary()
            self.net_op = tf.train.AdamOptimizer(learning_rate=0.001)

        else:
            # self.images = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3], name='Input')
            # self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, n_classes], name='labels')
            self.capsule_layer = capsule_layer(self)
            self.bulld_arch(num_classes=n_classes, iterations=1, name='capsulesEM-V0')
            self.get_accuracy()

        tf.logging.info('Seting up the main structure')

    def bulld_arch(self, num_classes=10, iterations=1, name='capsulesEM-V0'):

      with tf.variable_scope(name) as scope:

        # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
        nets = self.capsule_layer._conv2d_wrapper(
          inputs = self.images, shape=[5, 5, 3, 32], strides=[1, 1, 1, 1], padding='VALID', add_bias=True, activation_fn=tf.nn.relu, name='conv1'
        )


        # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
        nets = self.capsule_layer._conv2d_wrapper(
          inputs = nets, shape=[5, 5, 32, 32], strides=[1, 2, 2, 1], padding='SAME', add_bias=True, activation_fn=tf.nn.relu, name='conv2'
        )
        # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
        nets = self.capsule_layer.capsules_init(
          nets, shape=[1, 1, 32, 32], strides=[1, 1, 1, 1], padding='VALID', pose_shape=[4, 4], name='capsule_init'
        )
        # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 2 -> (poses, activations)
        nets = self.capsule_layer.capsules_conv(
          nets, shape=[3, 3, 32, 32], strides=[1, 2, 2, 1], iterations=iterations, name='capsule_conv1'
        )
        # inputs: (poses, activations) -> capsule-conv 3x3x32x32x4x4, strides 1 -> (poses, activations)
        nets = self.capsule_layer.capsules_conv(
          nets, shape=[3, 3, 32, 32], strides=[1, 1, 1, 1], iterations=iterations, name='capsule_conv2'
        )
        # inputs: (poses, activations) -> capsule-fc 1x1x32x10x4x4 shared view transform matrix within each channel -> (poses, activations)
        nets = self.capsule_layer.capsules_fc(
          nets, num_classes, iterations=iterations, name='capsule_fc'
        )

        poses, activations = nets
        self.logits = activations

    # ------------------------------------------------------------------------------#
    # ------------------------------------ loss ------------------------------------#
    # ------------------------------------------------------------------------------#

    def spread_loss(self,name='spread_loss'):
      """This adds spread loss to total loss.

      :param labels: [N, O], where O is number of output classes, one hot vector, tf.uint8.
      :param activations: [N, O], activations.
      :param margin: margin 0.2 - 0.9 fixed schedule during training.

      :return: spread loss
      """

      NUM_STEPS_PER_EPOCH = int(
        self.NUM_TRAIN_EXAMPLES / self.batch_size
      )
      margin_schedule_epoch_achieve_max = 10.0
      margin = tf.train.piecewise_constant(
        tf.cast(self.global_step, dtype=tf.int32),
        boundaries=[
          int(NUM_STEPS_PER_EPOCH * margin_schedule_epoch_achieve_max * x / 7) for x in range(1, 8)
        ],
        values=[
          x / 10.0 for x in range(2, 10)
        ]
      )
      activations_shape = self.logits.get_shape().as_list()

      with tf.variable_scope(name) as scope:

        mask_t = tf.equal(self.labels, 1)
        mask_i = tf.equal(self.labels, 0)

        activations_t = tf.reshape(
          tf.boolean_mask(self.logits, mask_t), [-1, 1]
        )
        activations_i = tf.reshape(
          tf.boolean_mask(self.logits, mask_i), [-1, activations_shape[1] - 1]
        )

        # margin = tf.Print(
        #   margin, [margin], 'margin', summarize=20
        # )

        gap_mit = tf.reduce_sum(
          tf.square(
            tf.nn.relu(
              margin - (activations_t - activations_i)
            )
          )
        )

        # tf.add_to_collection(
        #   tf.GraphKeys.LOSSES, gap_mit
        # )
        #
        # total_loss = tf.add_n(
        #   tf.get_collection(c
        #     tf.GraphKeys.LOSSES
        #   ), name='total_loss'
        # )

        # tf.losses.add_loss(gap_mit)
        # self.total_loss = tf.losses.get_total_loss()
        self.total_loss = gap_mit

    def get_accuracy(self):
        self.correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.labels, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def _summary(self):
      train_summary = []
      train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
      train_summary.append(tf.summary.scalar('train/accuracy', self.accuracy))
      self.merge_summary = tf.summary.merge_all()

      #self.train_summary = tf.summary.merge(train_summary)