from config import *
from net.cifarnet import cifarnet
from net.cifarnet import cifarnet_arg_scope
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


##################################   begin   ############################################
    input_images = tf.placeholder(dtype=tf.float32,shape = [None,32,32,3])
    input_labels = tf.placeholder(dtype=tf.float32,shape = [None,10])

    with slim.arg_scope(cifarnet_arg_scope()):
        logits, end_points = cifarnet(input_images, num_classes=10, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='CifarNet')



    total_loss = tf.reduce_sum(
      tf.nn.softmax_cross_entropy_with_logits(
        labels=input_labels, logits=logits, name='cross_entropy_loss'
      ))
    # )

    op_argmax_logits = tf.argmax(logits, axis=1)
    op_argmax_labels = tf.argmax(input_labels, axis=1)
    op_argmax_logits = tf.Print(op_argmax_logits, [op_argmax_logits], message='Debug logits', summarize=1000)
    op_argmax_labels = tf.Print(op_argmax_labels, [op_argmax_labels], message='Debug labels', summarize=1000)

    correct_prediction = tf.equal(op_argmax_logits, op_argmax_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



##################################   end   ############################################

    # TODO: set up a learning_rate decay
    # optimizer = tf.train.AdamOptimizer(
    #   learning_rate=0.001
    # )
    optimizer = tf.train.AdamOptimizer().minimize(total_loss, global_step=global_step)
    '''
    启动会话，开始训练
    '''
    training_epochs = 30
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


                # print('label : ',tf.argmax(input_labels, axis=1))



                _, loss,acc= sess.run([optimizer, total_loss,accuracy],
                                   feed_dict={input_images: input_train_x, input_labels: input_train_y})

                total_cost += loss
                print('global_step: ', step, ' loss: ', loss, '   acc: ', acc)


                if step[0] % 50 == 0:
                    print('summary.......! ')
                    merged = tf.summary.merge_all()
                    _,summary_str = sess.run([optimizer,merged],feed_dict={input_images: input_train_x, input_labels: input_train_y})
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