#importing dependencies
import tensorflow as tf
import numpy
# import pandas as pd
from data import *
#Helper functions to define weights and biases
def init_weights(shape):
    '''
    Input: shape -  this is the shape of a matrix used to represent weigts for the arbitrary layer
    Output: wights randomly generated with size = shape
    '''
    return tf.Variable(tf.truncated_normal(shape, 0, 0.05))

def init_biases(shape):
    '''
    Input: shape -  this is the shape of a vector used to represent biases for the arbitrary layer
    Output: a vector for biases (all zeros) lenght = shape
    '''
    return tf.Variable(tf.zeros(shape))


def fully_connected_layer(inputs, input_shape, output_shape, activation=tf.nn.relu):
    '''
    This function is used to create tensorflow fully connected layer.

    Inputs: inputs - input data to the layer
            input_shape - shape of the inputs features (number of nodes from the previous layer)
            output_shape - shape of the layer
            activatin - used as an activation function for the layer (non-liniarity)
    Output: layer - tensorflow fully connected layer

    '''
    # definine weights and biases
    weights = init_weights([input_shape, output_shape])
    biases = init_biases([output_shape])

    # x*W + b <- computation for the layer values
    layer = tf.matmul(inputs, weights) + biases

    # if activation argument is not None, we put layer values through an activation function
    if activation != None:
        layer = activation(layer)

    return layer

#%%
#splitting the dataset to the training set and the testing set
option_num = 0 # P V T options
class_num = 2
ford_num = 10
data, label = dataloader(class_num, option_num)
data, label = shuffle_2arr(data, label)
X_train, Y_train, X_test, Y_test = split_train_test(data, label, option_num, ford_num)

feature_num = len(X_train[0])
print(X_train.shape, )

#Tensorflow placeholders - inputs to the TF graph
inputs =  tf.placeholder(tf.float32, [None, feature_num], name='Inputs')
targets =  tf.placeholder(tf.float32, [None, class_num], name='Targets')

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

#defining the network
layer = [1024, 1024, 1024, 1024, 1024, 1024, 1024]
l1 = fully_connected_layer(inputs, feature_num, layer[0])
l2 = fully_connected_layer(l1, layer[0], layer[1]) + l1
l3 = fully_connected_layer(l2, layer[1], layer[2]) + l2
l4 = fully_connected_layer(l3, layer[2], layer[3]) + l3
l5 = fully_connected_layer(l4, layer[3], layer[4]) + l4
l6 = fully_connected_layer(l5, layer[4], layer[5]) + l5
l7 = fully_connected_layer(l6, layer[5], layer[6]) + l6
l8 = fully_connected_layer(l7, layer[6], class_num, activation=None)

#defining special parameter for our predictions - later used for testing
predictions = tf.nn.sigmoid(l8)

#Mean_squared_error function and optimizer choice - Classical Gradient Descent
cost = loss2 = tf.reduce_mean(tf.squared_difference(targets, predictions))
tf.summary.scalar("cost", cost)
merged_summary = tf.summary.merge_all()

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# epochs = 10000
epochs = 1000
# batch_size = 50
# from tqdm import tqdm

# Starting session for the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./my_graph')
    writer.add_graph(sess.graph)
    # TRAINING PORTION OF THE SESSION
    # one hot encoding
    Y_train = pd.get_dummies(Y_train)
    Y_train = np.array(Y_train)
    Y_test = pd.get_dummies(Y_test)
    Y_test = np.array(Y_test)
    for i in range(epochs):
        '''
        idx = np.random.choice(len(X_train), batch_size, replace=True)
        x_batch = X_train[idx, :]
        y_batch = Y_train[idx]
        y_batch = np.reshape(y_batch, (len(y_batch), 1))
        '''
        y_batch = Y_train
        x_batch = X_train

        summary, batch_loss, opt, preds_train = sess.run([merged_summary, cost, optimizer, predictions], \
                                                         feed_dict={inputs: x_batch, targets: y_batch})
        writer.add_summary(summary, global_step=i)
        if i % 100 == 0:
            print('='*50)
            print('epoch : ',i, '/',epochs)
            accur = accuracy(preds_train, Y_train)
            print("Training Accuracy (%): ", accur)
            print('batch loss : ',batch_loss)

            # TESTING PORTION OF THE SESSION
            preds = sess.run([predictions], feed_dict={inputs: X_test})
            # preds_nparray = np.squeeze(np.array(preds), 0)
            preds_nparray = np.squeeze(np.array(preds), 0)

            # print(preds_nparray.shape, Y_test.shape)
            # print(np.argmax(preds_nparray, 1))
            # print(np.argmax(Y_test, 1))
            # assert False
            print(preds_nparray[:3])
            # print(Y_test)
            accur = accuracy(preds_nparray, Y_test)
            print("Test Accuracy (%): ", accur)
        writer.close()