import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# hidden layer function
def add_layer(inputs, in_size, out_size, act_func = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias
        if act_func is None:
            outputs = Wx_plus_b
        else:
            outputs = act_func(Wx_plus_b)
        return outputs

# creat second order data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise =np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# creat model structure
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')
l1 = add_layer(xs, 1, 10, act_func = tf.nn.relu)
prediction = add_layer(l1, 10, 1, act_func = None)

# optimize trainning
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
            reduction_indices = [1]))
with tf.name_scope('train'):
    tr_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter('logs/', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)
#
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.scatter(x_data, y_data)
#plt.ion()
#plt.show()
#for i in range(1000):
#    sess.run(tr_step, feed_dict = {xs:x_data, ys:y_data})
#    if i % 50 == 0:
#        # print(sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))
#        try:
#            ax.lines.remove(lines[0])
#        except Exception:
#            pass
#        prediction_value = sess.run(prediction, feed_dict = {xs: x_data})
#        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
#        plt.pause(0.1)
