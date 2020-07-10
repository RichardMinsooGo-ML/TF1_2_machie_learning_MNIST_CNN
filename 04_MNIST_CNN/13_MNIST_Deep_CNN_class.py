import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# Use os.**** for high level CPU, it can delete some warning messages.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

tf.set_random_seed(777)  # reproducibility

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# hyper parameters
Alpha_Lr   = 0.001
N_EPISODES = 10
batch_size = 100

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

class _MODEL:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._BUILD_NETWORK()

    def _BUILD_NETWORK(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.keep_prob = tf.placeholder(tf.float32)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])
            # img 28x28x1 (black/white)
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # _LAY01_m ImgIn shape=(?, 28, 28, 1)
            W01_m       = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            CONV_01     = tf.nn.conv2d(X_img, W01_m, strides=[1, 1, 1, 1], padding='SAME')
            RELU_01     = tf.nn.relu(CONV_01)
            MAX_POOL_01 = tf.nn.max_pool(RELU_01, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            DROP_OUT_01 = tf.nn.dropout(MAX_POOL_01, keep_prob=self.keep_prob)

            # _LAY02_m ImgIn shape=(?, 14, 14, 32)
            W02_m       = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            CONV_02     = tf.nn.conv2d(DROP_OUT_01, W02_m, strides=[1, 1, 1, 1], padding='SAME')
            RELU_02     = tf.nn.relu(CONV_02)
            MAX_POOL_02 = tf.nn.max_pool(RELU_02, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            DROP_OUT_02 = tf.nn.dropout(MAX_POOL_02, keep_prob=self.keep_prob)

            # _LAY03_m ImgIn shape=(?, 7, 7, 64)
            W03_m       = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128)
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            CONV_03     = tf.nn.conv2d(DROP_OUT_02, W03_m, strides=[1, 1, 1, 1], padding='SAME')
            RELU_03     = tf.nn.relu(CONV_03)
            MAX_POOL_03 = tf.nn.max_pool(RELU_03, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            DROP_OUT_03 = tf.nn.dropout(MAX_POOL_03, keep_prob=self.keep_prob)

            FLATTEN_03  = tf.reshape(DROP_OUT_03, [-1, 128 * 4 * 4])

            # _LAY04_m FC 4x4x128 inputs -> 625 outputs
            W04_m       = tf.get_variable("W04_m", shape=[128 * 4 * 4, 625],
                                          initializer=tf.contrib.layers.xavier_initializer())
            B04_m       = tf.Variable(tf.random_normal([625]))
            RELU_04     = tf.nn.relu(tf.matmul(FLATTEN_03, W04_m) + B04_m)
            DROP_OUT_04 = tf.nn.dropout(RELU_04, keep_prob=self.keep_prob)

            # L5 Final FC 625 inputs -> 10 outputs
            W05_m       = tf.get_variable("W05_m", shape=[625, 10],
                                          initializer=tf.contrib.layers.xavier_initializer())
            B05_m       = tf.Variable(tf.random_normal([10]))
            self.Pred_m = tf.matmul(DROP_OUT_04, W05_m) + B05_m

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Pred_m, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.Pred_m, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _PREDICT(self, x_test, keep_prop=1.0):
        return self.sess.run(self.Pred_m, feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def _GET_ACCURACY(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def _TRAIN_NET(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

# initialize
sess = tf.Session()
m1 = _MODEL(sess, "m1")
init = tf.global_variables_initializer()
sess.run(init)

# train my model
print('Learning started. It takes sometime.')

# train my model
for episode in range(N_EPISODES):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1._TRAIN_NET(batch_xs, batch_ys)
        avg_cost += c / total_batch

    print('episode:', '%05d' % (episode + 1), 'cost =', '{:.5f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
print('Accuracy:', m1._GET_ACCURACY(mnist.test.images, mnist.test.labels))

#########
# 결과 확인 (matplot)
######
#labels = sess.run(Pred_m,feed_dict={X: mnist.test.images,Y: mnist.test.labels,keep_prob: 1})
#labels = m1.train(feed_dict={X: mnist.test.images,Y: mnist.test.labels,keep_prob: 1})
labels = m1._PREDICT(mnist.test.images,1)
fig = plt.figure()
for i in range(60):
    subplot = fig.add_subplot(4, 15, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()


# 세션을 닫습니다.
sess.close()


# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:

