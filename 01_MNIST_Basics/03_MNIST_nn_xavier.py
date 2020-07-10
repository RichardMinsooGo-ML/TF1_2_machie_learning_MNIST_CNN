import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random

# Use os.**** for high level CPU, it can delete some warning messages.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Step 1. Import or generate datasets:
# Step 2. Transform and normalize data:
# Step 3. Partition datasets into train, test, and validation sets:

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Step 4. Set algorithm parameters (hyperparameters):
learning_rate = 0.001
N_Input = 784
N_Classes = 10
H_SIZE_01 = 256
H_SIZE_02 = 255
N_EPISODES = 10
# 100, 200, 500, 1000, 2000, 5000. For MNIST 5000 might work. If memory is insufficient chosose small value
batch_size = 5000

# Step 5. Initialize variables and placeholders:
# input place holders
X = tf.placeholder(tf.float32, [None, N_Input])
Y = tf.placeholder(tf.float32, [None, N_Classes])

# weights & bias for nn layers
W01_m = tf.get_variable("W01_m", shape=[N_Input, H_SIZE_01],
                     initializer=tf.contrib.layers.xavier_initializer())
B01_m = tf.Variable(tf.random_normal([H_SIZE_01]))

W02_m = tf.get_variable("W02_m", shape=[H_SIZE_01, H_SIZE_02],
                     initializer=tf.contrib.layers.xavier_initializer())
B02_m = tf.Variable(tf.random_normal([H_SIZE_02]))

W_output = tf.get_variable("W_output", shape=[H_SIZE_02, N_Classes],
                     initializer=tf.contrib.layers.xavier_initializer())
B_output = tf.Variable(tf.random_normal([N_Classes]))

_LAY01_m = tf.nn.relu(tf.matmul(X, W01_m) + B01_m)
_LAY02_m = tf.nn.relu(tf.matmul(_LAY01_m, W02_m) + B02_m)
Pred_m = tf.matmul(_LAY02_m, W_output) + B_output

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_m, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#total_batch = 10*int(mnist.train.num_examples / batch_size)
total_batch = int(mnist.train.num_examples / batch_size)

for episode in range(N_EPISODES):
    avg_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('episode:', '%04d' % (episode + 1), 'cost =', '{:3.5f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))
"""
# Option 1 Plot: Get 60 sequentially and predict
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
"""
# Option 2 Plot: Get 60 in random and predict
labels = sess.run(Pred_m,
                  feed_dict={X: mnist.test.images,
                             Y: mnist.test.labels})

fig = plt.figure()
X_test = mnist.test.images
N_test_sample = int(X_test.shape[0])
permutations = np.random.permutation(N_test_sample)
for i in range(60):
    index = permutations[i]
    subplot = fig.add_subplot(4, 15, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[index]))
    subplot.imshow(mnist.test.images[index].reshape((28, 28)),
                   cmap=plt.cm.gray_r)

plt.show()

# 세션을 닫습니다.
sess.close()

# Step 10. Tune hyperparameters:
# Step 11. Deploy/predict new outcomes:

