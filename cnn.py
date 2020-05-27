import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# Define the model
def build_CNN_classifier(x):
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First Convolution Layer 
  # Using 5 x 5 x 32 kernel (28x28x1 -> 28x28x32)
  W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # First Pooling Layer
  # Using Max Pooling, Stride 2x2 (28x28x32 -> 14x14x32)
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Second Convolutional Layer 
  # Using 5 x 5 x 32 kernel (14x14x32 -> 14x14x64)
  W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # Second Pooling Layer
  # Using Max Pooling, stride 2x2 (14x14x64 -> 7x7x64)
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Fully Connected Layer
  # 7x7x64 -> 1024
  W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Output Layer
  # 1024개의 특징들(feature)을 10개의 클래스-숫자 0-9-로 변환합니다.
  # 1024 -> 10
  W_output = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=5e-2))
  b_output = tf.Variable(tf.constant(0.1, shape=[10]))
  logits = tf.matmul(h_fc1, W_output) + b_output
  y_pred = tf.nn.softmax(logits)

  return y_pred, logits

def train():
    # read datasets
    mnist = input_data.read_data_sets("./dataset/", one_hot=True)

    # Define the placeholder(x: input, y: class)
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # Declear Convolutional Neural Networks(CNN)
    y_pred, logits = build_CNN_classifier(x)

    # y_pred_class = tf.math.argmax(y_pred)

    # Define the optimizer using the loss from cross entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # Save the model
    SAVER_DIR = "model/ckpt/"
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, "model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    with tf.Session() as sess:
        # Initialize all variable
        sess.run(tf.global_variables_initializer())

        # # If using previous model, restore the model and test
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)    
        #     print("Test Accuracy (Restored) : %f" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))
        #     sess.close()
        #     exit()
        
        # Write logs
        writer = tf.summary.FileWriter("./logs/", sess.graph)

        for step in range(1000):
            # batch size 50
            batch = mnist.train.next_batch(50)
            # Calculate the train accuracy and save the model at each 100 steps
            if step % 100 == 0:      
                saver.save(sess, checkpoint_path, global_step=step)
                tf.io.write_graph(sess.graph_def, '.', './model/ckpt/model.pb', as_text=False)
                tf.io.write_graph(sess.graph_def, '.', './model/ckpt/model.pbtxt', as_text=True)
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
                print("Epoch: %d, Training Accuracy: %f" % (step, train_accuracy))
            # Training with optimizer
            sess.run([train_step], feed_dict={x: batch[0], y: batch[1]})

        # Evaluate about test dataset
        print("Test Accuracy: %f" % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))

if __name__ == "__main__":
    train()