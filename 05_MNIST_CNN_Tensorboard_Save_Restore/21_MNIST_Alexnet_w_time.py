import tensorflow as tf
import time, datetime
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
# 최신 Windows Laptop에서만 사용할것.CPU Version이 높을때 사용.
# AVX를 지원하는 CPU는 Giuthub: How to compile tensorflow using SSE4.1, SSE4.2, and AVX. 
# Ubuntu와 MacOS는 지원하지만 Windows는 없었음. 2018-09-29
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Compuntational Graph Initialization
from tensorflow.python.framework import ops
ops.reset_default_graph()

DATA_DIR = "/tmp/ML/MNIST_data"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Parameters
Alpha_Lr       = 0.001
training_iters = 20000
batch_size     = 100
display_step   = 20

# Network Parameters
INPUT_SIZE  = 784    # MNIST data input (img shape: 28*28)
OUTPUT_SIZE = 10     # MNIST total classes (0-9 digits)
dropout     = 0.7    # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Create AlexNet model
########## define conv process ##########
def CONVOLUTION(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

########## define pool process ##########
def POOLING(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

########## define norm process ##########
def NORMALIZATION(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

########## set net parameters ##########
def weight_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_var(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

weights={
    'WEI_CONV_01': weight_var('WEI_CONV_01',[3, 3, 1, 64]),
    'WEI_CONV_02': weight_var('WEI_CONV_02',[3, 3, 64, 128]),
    'WEI_CONV_03': weight_var('WEI_CONV_03',[3, 3, 128, 256]),
    'WEI_DENS_01': weight_var('WEI_DENS_01',[4*4*256, 1024]),
    'WEI_DENS_02': weight_var('WEI_DENS_02',[1024, 1024]),
    'out': weight_var('out',[1024, 10])
}
biases={
    'BIA_CONV_01': bias_var('BIA_CONV_01',[64]),
    'BIA_CONV_02': bias_var('BIA_CONV_02',[128]),
    'BIA_CONV_03': bias_var('BIA_CONV_03',[256]),
    'BIA_DENS_01': bias_var('BIA_DENS_01',[1024]),
    'BIA_DENS_02': bias_var('BIA_DENS_02',[1024]),
    'out': bias_var('out',[OUTPUT_SIZE])
}

"""
########## set net parameters ##########
weights = {
    'WEI_CONV_01': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'WEI_CONV_02': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'WEI_CONV_03': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'WEI_DENS_01': tf.Variable(tf.random_normal([4*4*256, 1024])),
    'WEI_DENS_02': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'BIA_CONV_01': tf.Variable(tf.random_normal([64])),
    'BIA_CONV_02': tf.Variable(tf.random_normal([128])),
    'BIA_CONV_03': tf.Variable(tf.random_normal([256])),
    'BIA_DENS_01': tf.Variable(tf.random_normal([1024])),
    'BIA_DENS_02': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([OUTPUT_SIZE]))
}
"""

def ALEX_NET(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])

    # Convolution Layer
    CONV_01     = CONVOLUTION('CONV_01', _X, _weights['WEI_CONV_01'], _biases['BIA_CONV_01'])
    # Max Pooling (down-sampling)
    MAX_POOL_01 = POOLING('MAX_POOL_01', CONV_01, k=2)
    # Apply Normalization
    NORM_01     = NORMALIZATION('NORM_01', MAX_POOL_01, lsize=4)
    # Apply Dropout
    NORM_01     = tf.nn.dropout(NORM_01, _dropout)

    # Convolution Layer
    CONV_02     = CONVOLUTION('CONV_02', NORM_01, _weights['WEI_CONV_02'], _biases['BIA_CONV_02'])
    # Max Pooling (down-sampling)
    MAX_POOL_02 = POOLING('MAX_POOL_02', CONV_02, k=2)
    # Apply Normalization
    NORM_02     = NORMALIZATION('NORM_02', MAX_POOL_02, lsize=4)
    # Apply Dropout
    NORM_02     = tf.nn.dropout(NORM_02, _dropout)

    # Convolution Layer
    CONV_03     = CONVOLUTION('CONV_03', NORM_02, _weights['WEI_CONV_03'], _biases['BIA_CONV_03'])
    # Max Pooling (down-sampling)
    MAX_POOL_03 = POOLING('MAX_POOL_03', CONV_03, k=2)
    # Apply Normalization
    NORM_03     = NORMALIZATION('NORM_03', MAX_POOL_03, lsize=4)
    # Apply Dropout
    NORM_03     = tf.nn.dropout(NORM_03, _dropout)

    # Fully connected layer
    DENSE_01 = tf.reshape(NORM_03, [-1, _weights['WEI_DENS_01'].get_shape().as_list()[0]]) # Reshape CONV_03 output to fit dense layer input
    DENSE_01 = tf.nn.relu(tf.matmul(DENSE_01, _weights['WEI_DENS_01']) + _biases['BIA_DENS_01'], name='fc1') # Relu activation

    DENSE_02 = tf.nn.relu(tf.matmul(DENSE_01, _weights['WEI_DENS_02']) + _biases['BIA_DENS_02'], name='fc2') # Relu activation

    # Output, class prediction
    Pred_m   = tf.matmul(DENSE_02, _weights['out']) + _biases['out']
    return Pred_m

# Construct model
pred = ALEX_NET(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = Alpha_Lr).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            
            elapsed_time = datetime.timedelta(seconds=int(time.time()-start_time))
            print("[{}]".format(elapsed_time))

        step += 1
    
    print("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
    
    elapsed_time = time.time() - start_time
    formatted = datetime.timedelta(seconds=int(elapsed_time))
    print("=== training time elapsed: {}s ===".format(formatted))
