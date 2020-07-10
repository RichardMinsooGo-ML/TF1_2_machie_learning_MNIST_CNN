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
# keep_prob = tf.placeholder(tf.float32)

W01_m = tf.Variable(tf.random_normal([N_Input, H_SIZE_01], stddev=0.01))
B01_m = tf.Variable(tf.random_normal([H_SIZE_01]))

W02_m = tf.Variable(tf.random_normal([H_SIZE_01, H_SIZE_02], stddev=0.01))
B02_m = tf.Variable(tf.random_normal([H_SIZE_02]))

W_output = tf.Variable(tf.random_normal([H_SIZE_02, N_Classes], stddev=0.01))
B_output = tf.Variable(tf.random_normal([N_Classes]))

# Step 6. Define the Pred_m structure:
_LAY01_m = tf.nn.relu(tf.matmul(X, W01_m)+B01_m)
_LAY02_m = tf.nn.relu(tf.matmul(_LAY01_m, W02_m)+B02_m)
Pred_m = tf.matmul(_LAY02_m, W_output)+B_output

# Step 7. Declare the loss functions:

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred_m, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

# Step 8. Initialize and train the Pred_m:
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for episode in range(N_EPISODES):
    avg_cost = 0

    for i in range(total_batch):
        # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
        # 지정한 크기만큼 학습할 데이터를 가져옵니다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('episode:', '%04d' % (episode + 1), 'cost =', '{:3.5f}'.format(avg_cost))

print('Learning Finished!')

#########
# 결과 확인
######

correct_prediction = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images,
                                   Y: mnist.test.labels}))

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

