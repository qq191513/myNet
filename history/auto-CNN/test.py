import tensorflow as tf



u_1 = tf.placeholder(tf.float32, [784, 784])
first_layer_u = tf.layers.dense(X_, n_params, activation=None,
                              kernel_initializer=u_1,
                              bias_initializer=tf.keras.initializers.he_normal())