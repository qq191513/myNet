"""A train script for matrix capsule with EM routing."""

from config import *

from class_capsules.class_capsnet import capsules_v0

def main(_):
    graph = tf.Graph()
    with graph.as_default():
        images, labels= get_data_set(main_directory = FLAGS.data_dir,name="train",batch_size= FLAGS.batch_size)
        # test_x,test_y= get_data_set(main_directory = FLAGS.data_dir,name="test",batch_size= FLAGS.batch_size)

        tf.logging.set_verbosity(tf.logging.INFO)




        capsNet = capsules_v0(images = images, labels = labels,logdir =FLAGS.train_dir,
                              NUM_TRAIN_EXAMPLES =50000,batch_size=FLAGS.batch_size,
                              is_training=FLAGS.is_training)




        train_tensor = slim.learning.create_train_op(
          capsNet.total_loss, capsNet.net_op, global_step=capsNet.global_step, clip_gradient_norm=4.0
        )




        slim.learning.train(
          train_tensor,
          logdir=FLAGS.train_dir,
          log_every_n_steps=100,
          save_summaries_secs=300,
          saver=tf.train.Saver(max_to_keep=5),
          save_interval_secs=300,
          # yg: add session_config to limit gpu usage and allow growth
          session_config=tf.ConfigProto(
            # device_count = {
            #   'GPU': 0
            # },
            gpu_options={
              'allow_growth': 1,
              # 'per_process_gpu_memory_fraction': 0.01
              'visible_device_list': '0'
            },
            allow_soft_placement=True,
            log_device_placement=False
          )
        )
    # init = tf.global_variables_initializer()
    # sess = tf.Session(
    #   config=tf.ConfigProto(
    #     # device_count = {
    #     #   'GPU': 0
    #     # },
    #     gpu_options={
    #       'allow_growth': 0,
    #       # 'per_process_gpu_memory_fraction': 0.01
    #       'visible_device_list': '0'
    #     },
    #     allow_soft_placement=True,
    #     log_device_placement=False
    #   )
    # )
    # sess.run(init)
    #
    # tf.train.start_queue_runners(sess=sess)
    #
    # for step in range(1001):
    #   print('step: ', step)
    #   _, loss_value = sess.run([train_op, loss])
    #   print('step: ', step, 'loss: ', loss_value)
    #   if step % 10 == 0:
    #     print('gv')
    #     for gv in grads_and_vars:
    #       print(gv[1])
    #       print(str(sess.run(gv[1].name)))
    #       z = [0 for x in gv[0].shape]
    #       print(sess.run(gv[0][z]))


if __name__ == "__main__":
  tf.app.run()

