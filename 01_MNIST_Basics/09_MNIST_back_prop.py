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

W01_m = tf.Variable(tf.truncated_normal([N_Input, H_SIZE_01]))
B01_m = tf.Variable(tf.truncated_normal([1, H_SIZE_01]))

W_output = tf.Variable(tf.truncated_normal([H_SIZE_01, N_Classes]))
B_output = tf.Variable(tf.truncated_normal([1, N_Classes]))


def sigma(x):
    #  sigmoid function
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(-x)))


def sigma_prime(x):
    # derivative of the sigmoid function
    return sigma(x) * (1 - sigma(x))

# Forward prop
l1 = tf.add(tf.matmul(X, W01_m), B01_m)
a1 = sigma(l1)
l2 = tf.add(tf.matmul(a1, W_output), B_output)
Pred_m = sigma(l2)

# diff
assert Pred_m.shape.as_list() == Y.shape.as_list()
diff = (Pred_m - Y)


# Back prop (chain rule)
d_l2 = diff * sigma_prime(l2)
d_B_output = d_l2
d_W_output = tf.matmul(tf.transpose(a1), d_l2)

d_a1 = tf.matmul(d_l2, tf.transpose(W_output))
d_l1 = d_a1 * sigma_prime(l1)
d_B01_m = d_l1
d_W01_m = tf.matmul(tf.transpose(X), d_l1)


# Updating network using gradients
step = [
    tf.assign(W01_m, W01_m - learning_rate * d_W01_m),
    tf.assign(B01_m, B01_m - learning_rate *
              tf.reduce_mean(d_B01_m, reduction_indices=[0])),
    tf.assign(W_output, W_output - learning_rate * d_W_output),
    tf.assign(B_output, B_output - learning_rate *
              tf.reduce_mean(d_B_output, reduction_indices=[0]))
]

# 7. Running and testing the training process
acct_mat = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict={X: batch_xs,
                              Y: batch_ys})
    if i % 1000 == 0:
        res = sess.run(acct_res, feed_dict={X: mnist.test.images[:1000],
                                            Y: mnist.test.labels[:1000]})
        print(res)

# 8. Automatic differentiation in TensorFlow
cost = diff * diff
step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

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

