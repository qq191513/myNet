from config import *
from capsules.nets import capsules_v0
from capsules.nets import spread_loss

import os
def main(_):

  if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)

  with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    train_x, train_y= get_data_set(main_directory = FLAGS.data_dir,name="train",batch_size= FLAGS.batch_size)
    test_x,test_y= get_data_set(main_directory = FLAGS.data_dir,name="test",batch_size= FLAGS.batch_size)

    train_x = tf.reshape(train_x, [-1, 32, 32, 3], name='input_train')
    test_x = tf.reshape(test_x, [-1, 32, 32, 3], name='input_test')



    NUM_STEPS_PER_EPOCH = int(
       50000/ FLAGS.batch_size
      )



    with tf.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()

    input_images = tf.placeholder(dtype=tf.float32,shape = [None,32,32,3])
    input_labels = tf.placeholder(dtype=tf.float32,shape = [None,10])
##################################   begin   ############################################
    poses, activations = capsules_v0(
        input_images, num_classes=10, iterations=1, name='capsulesEM-V0'
    )
    # activations = tf.Print(
    #   activations, [activations.shape, activations[0, ...]], 'activations', summarize=20
    # )

    # inverse_temperature = tf.train.piecewise_constant(
    #   tf.cast(global_step, dtype=tf.int32),
    #   boundaries=[
    #     int(NUM_STEPS_PER_EPOCH * 10),
    #     int(NUM_STEPS_PER_EPOCH * 20),
    #     int(NUM_STEPS_PER_EPOCH * 30),
    #     int(NUM_STEPS_PER_EPOCH * 50),
    #   ],
    #   values=[0.001, 0.001, 0.002, 0.002, 0.005]
    # )

    # margin schedule
    # margin increase from 0.2 to 0.9 after margin_schedule_epoch_achieve_max
    margin_schedule_epoch_achieve_max = 10.0
    # obser_boundaries = [
    #     int(NUM_STEPS_PER_EPOCH * margin_schedule_epoch_achieve_max * x / 7) for x in range(1, 8)
    # ]
    #
    # obser_values = [
    #     x / 10.0 for x in range(2, 10)
    # ]

    margin = tf.train.piecewise_constant(
      tf.cast(global_step, dtype=tf.int32),
      boundaries=[
        int(NUM_STEPS_PER_EPOCH * margin_schedule_epoch_achieve_max * x / 7) for x in range(1, 8)
      ],
      values=[
        x / 10.0 for x in range(2, 10)
      ]
    )

    # loss = tf.reduce_sum(
    #   tf.nn.softmax_cross_entropy_with_logits(
    #     labels=labels, logits=activations, name='cross_entropy_loss'
    #   )
    # )

    correct_prediction = tf.equal(tf.argmax(activations, axis=1), tf.argmax(input_labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    total_loss = spread_loss(
        input_labels, activations, margin=margin, name='spread_loss'
    )

##################################   end   ############################################

    # TODO: set up a learning_rate decay
    # optimizer = tf.train.AdamOptimizer(
    #   learning_rate=0.001
    # )
    optimizer = tf.train.AdamOptimizer().minimize(total_loss, global_step=global_step)
    '''
    启动会话，开始训练
    '''
    training_epochs = 5
    display_epoch = 1

    saver = tf.train.Saver()
    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        try:
            print("\nTrying to restore last checkpoint ...")
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=FLAGS.train_dir)
            saver.restore(sess, save_path=last_chk_path)
            print("Restored checkpoint from:", last_chk_path)
        except ValueError:
            print("\nFailed to restore checkpoint. Initializing variables instead.")
            sess.run(tf.global_variables_initializer())

        tf.summary.scalar('train/accuracy', accuracy)
        tf.summary.scalar(
            'train/total_loss', total_loss
        )
        # sess.run(tf.global_variables_initializer())



        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()

        # 启动QueueRunner, 此时文件名才开始进队。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('开始训练!')
        for epoch in range(training_epochs):
            total_cost = 0.0

            #训练
            for i in range(NUM_STEPS_PER_EPOCH):

                input_train_x, input_train_y = sess.run([train_x, train_y])
                # if epoch % display_epoch == 0:

                step= sess.run([global_step])



                _, loss,acc= sess.run([optimizer, total_loss,accuracy],
                                   feed_dict={input_images: input_train_x, input_labels: input_train_y})
                total_cost += loss
                print('global_step: ', step, ' loss: ', loss, '   acc: ', acc)


                if step[0] % 50 == 0:
                    print('summary.......! ')
                    merged = tf.summary.merge_all()
                    _,summary_str = sess.run([optimizer,merged],feed_dict={input_images: input_train_x,input_labels: input_train_y})
                    train_writer.add_summary(summary_str, step[0])
                    print('summary finished ! ')

            # 训练平均cost
            print('Epoch {}/{}  Train average cost {:.9f}'.format(epoch + 1, training_epochs,total_cost / NUM_STEPS_PER_EPOCH))

            #用测试数据的准确率
            Number_of_tests = 20
            test_accuracy_value_total = 0
            for i in range(Number_of_tests):
                input_test_x, input_test_y = sess.run([test_x, test_y])
                _, test_accuracy_value = sess.run([optimizer, accuracy],
                                             feed_dict={input_images: input_test_x,
                                                        input_labels: input_test_y})
                test_accuracy_value_total += test_accuracy_value
            print('Epoch {}/{} '.format(epoch + 1, training_epochs), '准确率:',  test_accuracy_value_total/Number_of_tests)


            # 保存模型
            print('saving model...............')

            saver.save(sess, save_path=FLAGS.train_dir + 'model_epoch_{}'.format(epoch),
                       global_step=global_step)
            print('model saved ! ')

                ##################观察 begin
                # _, loss,acc,step,acti= sess.run([optimizer, total_loss,accuracy,global_step,activations],
                #                    feed_dict={input_images: input_train_x, input_labels: input_train_y})
                # total_cost += loss
                # acti_argmax = sess.run(tf.argmax(acti, axis=1))
                # input_train_y_argmax = sess.run(tf.argmax(input_train_y, axis=1))
                # predi = sess.run(tf.equal(acti_argmax,input_train_y_argmax ))
                # predi_result = sess.run(tf.reduce_mean(tf.cast(predi, tf.float32)))
                # print('global_step: ', step, ' loss: ', loss, '   acc: ', acc,'   predi_result: ', predi_result)
                # obser_margin = sess.run(margin)
                ##################观察 end

                #print('global_step: ', step, ' loss: ', loss, '   acc: ', acc )
                # 打印信息



        print('训练完成')
        # 终止线程
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    tf.app.run()