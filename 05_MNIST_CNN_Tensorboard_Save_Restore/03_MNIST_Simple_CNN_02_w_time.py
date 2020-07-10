import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
from tensorflow.examples.tutorials.mnist import input_data
import random
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

# Define Hyper Parameters
Alpha_Lr   = 0.001     # Learning Rate Alpha
N_EPISODES = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])
# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# _LAY01_m ImgIn shape=(?, 28, 28, 1)
W01_m = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
CONV_01     = tf.nn.conv2d(X_img, W01_m, strides=[1, 1, 1, 1], padding='SAME')
RELU_01     = tf.nn.relu(CONV_01)
MAX_POOL_01 = tf.nn.max_pool(RELU_01, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')

# _LAY02_m ImgIn shape=(?, 14, 14, 32)
W02_m = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
CONV_02     = tf.nn.conv2d(MAX_POOL_01, W02_m, strides=[1, 1, 1, 1], padding='SAME')
RELU_02     = tf.nn.relu(CONV_02)
DROP_OUT_02 = tf.nn.max_pool(RELU_02, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
FLATTEN_02  = tf.reshape(DROP_OUT_02, [-1, 7 * 7 * 64])

# Final FC 7x7x64 inputs -> 10 outputs
W03_m = tf.get_variable("W03_m", shape=[7 * 7 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
B03_m = tf.Variable(tf.random_normal([10]))
Pred_m = tf.matmul(FLATTEN_02, W03_m) + B03_m

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=Pred_m, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = time.time()

# train my model
print('Learning started. It takes sometime.')
for episode in range(N_EPISODES):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('episode:', '%05d' % (episode + 1), 'cost =', '{:.5f}'.format(avg_cost))
    
    elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
    print("[{}]".format(elapsed_time))
    
print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))
'''
# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(Pred_m, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

plt.imshow(mnist.test.images[r:r + 1].
          reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
'''

elapsed_time = time.time() - start_time
formatted = datetime.timedelta(seconds=int(elapsed_time))
print("=== training time elapsed: {}s ===".format(formatted))

#########
# 결과 확인 (matplot)
######
labels = sess.run(Pred_m,
                  feed_dict={X: mnist.test.images,
                             Y: mnist.test.labels})

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
