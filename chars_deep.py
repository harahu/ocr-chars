import tensorflow as tf
import os
import chars_dataset

# Disable compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model save path
M_PATH = os.path.join(os.getcwd(), "deep_model")



def _deepnn(x):
    """deepnn builds the graph for a deep net for classifying letters.

    Args:
        x: an input tensor with the dimensions (N_examples, 400), where
        400 is the number of pixels in a standard CHARS image.

    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 26), with values
        equal to the logits of classifying the digit into one of 26 classes (the
        letters a-z). keep_prob is a scalar placeholder for the probability of
        dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 20, 20, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = _weight_variable([5, 5, 1, 32])
    b_conv1 = _bias_variable([32])
    h_conv1 = tf.nn.relu(_conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = _max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = _weight_variable([5, 5, 32, 64])
    b_conv2 = _bias_variable([64])
    h_conv2 = tf.nn.relu(_conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = _max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 20x20 image
    # is down to 5x5x64 feature maps -- maps this to 1024 features.
    W_fc1 = _weight_variable([5 * 5 * 64, 1024])
    b_fc1 = _bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 5*5*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = _weight_variable([1024, 26])
    b_fc2 = _bias_variable([26])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def _conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def _weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def _bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def classify_chars(images):
    # Create the model
    x = tf.placeholder(tf.float32, [None, 400])

    # Build the graph for the deep net
    y_conv, keep_prob = _deepnn(x)

    # Define classification
    letter_class = tf.argmax(y_conv, 1)
    confidence = tf.maximum(y_conv, 1)

    # Enable saving and loading of variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, M_PATH)
        print("Model restored.")

        images_classes = letter_class.eval(feed_dict={x: images, keep_prob: 1.0})
        images_confidence = confidence.eval(feed_dict={x: images, keep_prob: 1.0})

        return images_classes, images_confidence


def train_model():
    # Import data
    chars = chars_dataset.read_data_sets()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 400])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 26])

    # Build the graph for the deep net
    y_conv, keep_prob = _deepnn(x)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Enable saving and loading of variables
    saver = tf.train.Saver()
    if not os.path.exists(M_PATH):
        os.mkdir(M_PATH)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        max_ta = 0.80
        for i in range(20000):
            batch = chars.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_accuracy = accuracy.eval(feed_dict={
                    x: chars.test.images, y_: chars.test.labels, keep_prob: 1.0})
                print('step {}, training accuracy {:.2f}, test accuracy {:.2f}'.format(
                    i, train_accuracy, test_accuracy))
                if test_accuracy > max_ta:
                    # Save the variables to disk.
                    save_path = saver.save(sess, os.path.join(M_PATH, "chess_model.ckpt"))
                    print("Model saved in file: {}".format(save_path))
                    max_ta = test_accuracy
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy {:.2f}'.format(accuracy.eval(feed_dict={
            x: chars.test.images, y_: chars.test.labels, keep_prob: 1.0})))

if __name__ == '__main__':
    train_model()
