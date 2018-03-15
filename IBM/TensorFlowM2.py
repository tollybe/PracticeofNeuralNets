
import numpy as np
from scipy import signal as sg
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

"""
x= [6,2]
h= [1,2,5,4]

#set parameter to "full"
y= np.convolve(x,h,"full")  #zero padding = final dimension array bigger than Y

#parameter to "same"
y= np.convolve(x,h,"same")  #zero padding, =  same array as y

#parameter as " valid"
y= np.convolve(x,h,"valid")  #no padding = dimensionality reduction



I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]​
g= [[-1,1]]

print ('Without zero padding \n')
#consists of elements that do not rely on zero padding
print ('{0} \n'.format(sg.convolve( I, g, 'valid')))

print ('With zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'full')))
# The output is the full discrete linear convolution of the inputs.
# It will use zero to complete the input matrix

print ('With zero padding_same_ \n')
print ('{0} \n'.format(sg.convolve( I, g, 'same')))
# The output is the full discrete linear convolution of the inputs.
# It will use zero to complete the input matrix

#Building graph

input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

#Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)

# download standard image
bird = "https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg"

raw= input()

im = Image.open(raw)

image_gr = im.convert("L")
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)

image_gr = im.convert("L")
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)

# Plot image

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

#edge detector kernel
kernel = np.array([
                        [ 0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0],
                                     ])

grad = sg.convolve2d(arr, kernel, mode='same', boundary='symm')

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

##MNIST DATASET - Convolutional NN

#label for a specific digit
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot= True)


session = tf.InteractiveSession()
#placeholders for inputs and outputs

x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight tensor
W = tf.Variable(tf.zeros([784,10],tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32))

# run the op initialize_all_variables using an interactive session
session.run(tf.initialize_all_variables())
#mathematical operation to add weights and biases to the inputs
tf.matmul(x,W) + b

#softmax Regression
y = tf.nn.softmax(tf.matmul(x,W) + b)

#cost function - minibatch Gradient descent
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#train using costfunction
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Load 50 training examples for each training iteration
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100
print("The final accuracy for the simple ANN model is: {} % ".format(acc) )

session.close() #finish the session

#evaluation? The final accuracy for the simple ANN model is: 91.2%
#to improve: use a simple deep neural network with dropout
"""

### UPGRADE TO PREVIOUS MODEL  ###

session = tf.InteractiveSession()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot= True)


width = 28
height = 28
flat = width
class_output = 10

x = tf.placeholder(tf.float32, shape=[None, flat])
y_ = tf.placeholder(tf.float32, shape=[None, class_output])

x_image = tf.reshape(x, [-1,28,28,1])
x_image

#kernals weight and bias
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

#tensor of shape [batch_height, width, channels]
convolve1= tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1

#relu activation function
h_conv1 = tf.nn.relu(convolve1)

#max pooling
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2 conv1

#apply layer numero dos
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

#Convolve image with weight tensor and add biases.
convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')+ b_conv2

#Apply the ReLU activation Function
h_conv2 = tf.nn.relu(convolve2)
#Apply the max pooling
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2 conv2

#connect softmax to flat array
layer2_matrix = tf.reshape(conv2, [-1, 7*7*64])

#feature map
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs
#Matrix Multiplication (applying weights and biases)
fcl=tf.matmul(layer2_matrix, W_fc1) + b_fc1

#Apply the ReLU activation Function
h_fc1 = tf.nn.relu(fcl)
h_fc1

#Reduce overfitting/ feature selection
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)
layer_drop

#Readout layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

#Matrix multiplication
fc=tf.matmul(layer_drop, W_fc2) + b_fc2

#apply softmax activation function
y_CNN= tf.nn.softmax(fc)
y_CNN

#define functions and train model

#loss function
layer4_test =[[0.9, 0.1, 0.1],[0.9, 0.1, 0.1]]
y_test=[[1.0, 0.0, 0.0],[1.0, 0.0, 0.0]]
np.mean( -np.sum(y_test * np.log(layer4_test),1))

#cross entropy to measure error at softmax layer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), reduction_indices=[1]))

#optimiser
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#prediction
correct_prediction = tf.equal(tf.argmax(y_CNN,1), tf.argmax(y_,1))

#accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run session
session.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, float(train_accuracy)))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#evaluate
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
