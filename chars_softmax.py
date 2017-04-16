import tensorflow as tf
import sys, os
import chars_dataset

# Disable compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import data
chars = chars_dataset.read_data_sets()

# Create the model
x = tf.placeholder(tf.float32, [None, 400])
W = tf.Variable(tf.zeros([400, 26]))
b = tf.Variable(tf.zeros([26]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 26])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    # Initialise variables and start the session
    sess.run(init)

    # Train
    for epoch in range(10000):
        print('Epoch {}'.format(epoch))
        batch_xs, batch_ys = chars.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))

    # Test trained model
    print('Test set')
    test_xs, test_ys = (chars.test.images, chars.test.labels)
    print(sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}))

del sess
sys.exit(0)
