import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
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

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._BUILD_NETWORK()

    def _BUILD_NETWORK(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1
            # for testing
            self.training = tf.placeholder(tf.bool)

            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 784])

            # img 28x28x1 (black/white), Input Layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            CONV_01     = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                           padding="SAME", activation=tf.nn.relu)
            # Pooling Layer #1
            MAX_POOL_01 = tf.layers.max_pooling2d(inputs=CONV_01, pool_size=[2, 2],
                                                  padding="SAME", strides=2)
            DROP_OUT_01 = tf.layers.dropout(inputs=MAX_POOL_01,
                                            rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            CONV_02     = tf.layers.conv2d(inputs=DROP_OUT_01, filters=64, kernel_size=[3, 3],
                                           padding="SAME", activation=tf.nn.relu)
            MAX_POOL_02 = tf.layers.max_pooling2d(inputs=CONV_02, pool_size=[2, 2],
                                                  padding="SAME", strides=2)
            DROP_OUT_02 = tf.layers.dropout(inputs=MAX_POOL_02,
                                            rate=0.7, training=self.training)

            # Convolutional Layer #2 and Pooling Layer #2
            CONV_03     = tf.layers.conv2d(inputs=DROP_OUT_02, filters=128, kernel_size=[3, 3],
                                     padding="SAME", activation=tf.nn.relu)
            MAX_POOL_03 = tf.layers.max_pooling2d(inputs=CONV_03, pool_size=[2, 2],
                                            padding="SAME", strides=2)
            DROP_OUT_03 = tf.layers.dropout(inputs=MAX_POOL_03,
                                         rate=0.7, training=self.training)

            # Dense Layer with Relu
            flat = tf.reshape(DROP_OUT_03, [-1, 128 * 4 * 4])
            dense4 = tf.layers.dense(inputs=flat,
                                     units=625, activation=tf.nn.relu)
            DROP_OUT_04 = tf.layers.dropout(inputs=dense4,
                                         rate=0.7, training=self.training)

            # Pred_m (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.Pred_m = tf.layers.dense(inputs=DROP_OUT_04, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.Pred_m, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.Pred_m, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.Pred_m,feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,feed_dict={self.X: x_test,self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X: x_data, self.Y: y_data, self.training: training})

# initialize
sess = tf.Session()

models = []
num_models = 5
for m in range(num_models):
    models.append(Model(sess, "model" + str(m)))

sess.run(tf.global_variables_initializer())

print('Learning Started!')

# train my model
for episode in range(N_EPISODES):
    avg_cost_list = np.zeros(len(models))
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # train each model
        for m_idx, m in enumerate(models):
            c, _ = m.train(batch_xs, batch_ys)
            avg_cost_list[m_idx] += c / total_batch

    print('episode:', '%05d' % (episode + 1), 'cost =',avg_cost_list)
    
print('Learning Finished!')
          
# Test model and check accuracy
test_size = len(mnist.test.labels)
predictions = np.zeros([test_size, 10])
for m_idx, m in enumerate(models):
    print(m_idx, 'Accuracy:', m.get_accuracy(
        mnist.test.images, mnist.test.labels))
    p = m.predict(mnist.test.images)
    predictions += p

ensemble_correct_prediction = tf.equal(
    tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
ensemble_accuracy = tf.reduce_mean(
    tf.cast(ensemble_correct_prediction, tf.float32))
print('Ensemble accuracy:', sess.run(ensemble_accuracy))

#########
# 결과 확인 (matplot)
######
#labels = sess.run(Pred_m,feed_dict={X: mnist.test.images,Y: mnist.test.labels,keep_prob: 1})
#labels = m1.train(feed_dict={X: mnist.test.images,Y: mnist.test.labels,keep_prob: 1})
labels = m.predict(mnist.test.images,1)
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

