# Lab 10 MNIST and Deep learning
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, datetime
from tensorflow.examples.tutorials.mnist import input_data
import random

# Use os.**** for high level CPU, it can delete some warning messages.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

DATA_DIR = "/tmp/ML/MNIST_data"

# Step 1. Import or generate datasets:
# Step 2. Transform and normalize data:

tf.set_random_seed(1234)  # reproducibility

mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Step 4. Set algorithm parameters (hyperparameters):
learning_rate = 0.001
N_EPISODES = 100
batch_size = 100

H_SIZE_01 = 256
H_SIZE_02 = 255 
H_SIZE_03 = 257
H_SIZE_04 = 258

DIR_Checkpoint  = "/tmp/ML/08_MNIST_Tensorboard_Save_Restore/CheckPoint"
DIR_Tensorboard = "/tmp/ML/08_MNIST_Tensorboard_Save_Restore/Tensorboard"

# 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
global_step = tf.Variable(0, trainable=False, name='global_step')

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 9.	Define the model structure – Main Network

with tf.name_scope('Hidden_Layer_01'):
    W01_m = tf.get_variable("W01_m", shape=[784, H_SIZE_01],
                         initializer=tf.contrib.layers.xavier_initializer())
    B01_m = tf.Variable(tf.random_normal([H_SIZE_01]))
    _LAY01_m = tf.nn.relu(tf.matmul(X, W01_m) + B01_m)
    
with tf.name_scope('Hidden_Layer_02'):
    W02_m = tf.get_variable("W02_m", shape=[H_SIZE_01, H_SIZE_02],
                         initializer=tf.contrib.layers.xavier_initializer())
    B02_m = tf.Variable(tf.random_normal([H_SIZE_02]))
    _LAY02_m = tf.nn.relu(tf.matmul(_LAY01_m, W02_m) + B02_m)

with tf.name_scope('Hidden_Layer_03'):
    W03_m = tf.get_variable("W03_m", shape=[H_SIZE_02, H_SIZE_03],
                         initializer=tf.contrib.layers.xavier_initializer())
    B03_m = tf.Variable(tf.random_normal([H_SIZE_03]))
    _LAY03_m = tf.nn.relu(tf.matmul(_LAY02_m, W03_m) + B03_m)

with tf.name_scope('Hidden_Layer_04'):
    W04_m = tf.get_variable("W04_m", shape=[H_SIZE_03, H_SIZE_04],
                         initializer=tf.contrib.layers.xavier_initializer())
    B04_m = tf.Variable(tf.random_normal([H_SIZE_04]))
    _LAY04_m = tf.nn.relu(tf.matmul(_LAY03_m, W04_m) + B04_m)

with tf.name_scope('Output_Layer'):
    W05_m = tf.get_variable("W05_m", shape=[H_SIZE_04, 10],
                         initializer=tf.contrib.layers.xavier_initializer())
    B05_m = tf.Variable(tf.random_normal([10]))
    Pred_m = tf.matmul(_LAY04_m, W05_m) + B05_m
    
with tf.name_scope('optimizer'):
    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=Pred_m, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
    
    # tf.summary.scalar 를 사용하여 기록해야할 tensor들을 수집, tensor들이 너무 많으면 시각화가 복잡하므로 간단한것만 선택.
    tf.summary.scalar('cost', cost)

# initialize
init = tf.global_variables_initializer()
sess = tf.Session()
start_time = time.time()

if not os.path.exists(DIR_Checkpoint):
    os.makedirs(DIR_Checkpoint)
if not os.path.exists(DIR_Tensorboard):
    os.makedirs(DIR_Tensorboard)
    
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state(DIR_Checkpoint)

if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Variables are restored!')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
else:
    sess.run(init)
    print('Variables are initialized!')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
# 텐서보드에서 표시해주기 위한 텐서들을 수집합니다.
merged = tf.summary.merge_all()
# 저장할 그래프와 텐서값들을 저장할 디렉토리를 설정합니다.
writer = tf.summary.FileWriter(DIR_Tensorboard, sess.graph)
# 이렇게 저장한 로그는, 학습 후 다음의 명령어를 이용해 웹서버를 실행시킨 뒤
# tensorboard --logdir=./logs
# 다음 주소와 웹브라우저를 이용해 텐서보드에서 확인할 수 있습니다.
# http://localhost:6006

# train my model
print('Learning started. It takes sometime.')

# train my model
for episode in range(N_EPISODES):
    avg_cost = 0
    total_batch = 50
    # total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Global Step:', '%07d' % int(sess.run(global_step)/total_batch), 'cost =', '{:02.6f}'.format(avg_cost))
    
    elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
    print("[{}]".format(elapsed_time))

    # 적절한 시점에 저장할 값들을 수집하고 저장합니다.
    summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys})
    writer.add_summary(summary, global_step=sess.run(global_step))

print('Learning Finished!')

# 최적화가 끝난 뒤, 변수를 저장합니다.
#saver.save(sess, './model/dnn.ckpt', global_step=global_step)

saver.save(sess, DIR_Checkpoint + './dnn.ckpt', global_step=global_step)

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(Pred_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Test Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

elapsed_time = time.time() - start_time
formatted = datetime.timedelta(seconds=int(elapsed_time))
print("=== training time elapsed: {}s ===".format(formatted))

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

