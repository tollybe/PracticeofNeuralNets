##recurrent networks in deep learning##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tarfile
import codecs
import os
import collections
from six.moves import cPickle
import time
#from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import seq2seq


"""
sess = tf.Session()

LSTM_CELL_SIZE = 4  # output size (dimension)

lstm_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([2,LSTM_CELL_SIZE]),)*2
state

sample_input = tf.constant([[1,2,3,4,3,2],[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))

print (sess.run(output))

###stacked LSTM Basic###

sess = tf.Session()
LSTM_CELL_SIZE = 4  #4 hidden nodes = state_dim = the output_dim
input_dim = 6
num_layers = 2

#stacked LSTM cell
cells = []
for _ in range(num_layers):
    cell = tf.contrib.rnn.LSTMCell(LSTM_CELL_SIZE)
    cells.append(cell)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(cells)

# Batch size x time steps x features.
data = tf.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

#Batch size x time steps x features.
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input

sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})


##LSTM for Classification using MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)

trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels

ntrain = trainimgs.shape[0]
ntest = testimgs.shape[0]
dim = trainimgs.shape[1]
nclasses = trainlabels.shape[1]
print ("Train Images: ", trainimgs.shape)
print ("Train Labels  ", trainlabels.shape)
print ()
print ("Test Images:  " , testimgs.shape)
print ("Test Labels:  ", testlabels.shape)

samplesIdx = [300, 409, 78]

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(testimgs[samplesIdx[0]].reshape([28,28]), cmap='gray')


xx, yy = np.meshgrid(np.linspace(0,28,28), np.linspace(0,28,28))
X =  xx ; Y =  yy
Z =  100*np.ones(X.shape)

img = testimgs[77].reshape([28,28])
ax = fig.add_subplot(122, projection='3d')
ax.set_zlim((0,200))


offset=200
for i in samplesIdx:
    img = testimgs[i].reshape([28,28]).transpose()
    ax.contourf(X, Y, img, 200, zdir='z', offset=offset, cmap="gray")
    offset -= 100

    ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()


for i in samplesIdx:
    print ("Sample: {0} - Class: {1} - Label Vector: {2} ".format(i, np.nonzero(testlabels[i])[0], testlabels[i]))

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

#construct x, y values for recurrent neural network
x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x") # Current data input shape: (batch_size, n_steps, n_input) [100x28x28]
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")

#w + b for read out layer
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#dynamic_rnn = recurrent neural network specified from lstm_cell
lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)


outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)

#reshape output + prediction
output = tf.reshape(tf.split(outputs, 28, axis=1, num=None, name='split')[-1],[-1,128])
pred = tf.matmul(output, weights['out']) + biases['out']

#labels + logits should be tensors of shape [100 x 10]
pred

#cost function + optimiser
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred ))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#accuracy + evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        # We will read a batch of 100 images [100 x 784] as batch_x
        # batch_y is a matrix of [100x10]
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # We consider each row of the image as one sequence
        # Reshape data to get 28 seq of 28 elements, so that, batxh_x is [100x28x28]
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

sess.close()

"""
"""
##LANGUAGE MODEL - LSTM ##
import os, sys

data = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
#Note to self: work out why it's not extracting file

tar = tarfile.r(data)
tar.getmembers()
os.chdir("/tmp/foo")
for member in tar.getmembers():
    f = tar.extractfile(member)
    content = f.read()
    sys.exit
tar.close()
print ("complete")

#Initial weight scale
init_scale = 0.1
#Initial learning rate
learning_rate = 1.0
#Maximum permissible norm for the gradient (For gradient clipping -- another measure against Exploding Gradients)
max_grad_norm = 5
#The number of layers in our model
num_layers = 2
#The total number of recurrence steps, also known as the number of layers when our RNN is "unfolded"
num_steps = 20
#The number of processing units (neurons) in the hidden layers
hidden_size = 200
#The maximum number of epochs trained with the initial learning rate
max_epoch = 4
#The total number of epochs in training
max_max_epoch = 13
#The probability for keeping data in the Dropout Layer (This is an optimization, but is outside our scope for this notebook!)
#At 1, we ignore the Dropout Layer wrapping.
keep_prob = 1
#The decay for the learning rate
decay = 0.5
#The size for each batch of data
batch_size = 30
#The size of our vocabulary
vocab_size = 10000
#Training flag to separate training from testing
is_training = 1
#Data directory for our dataset
data_dir = "/resources/data/simple-examples/data/"

session=tf.InteractiveSession()

# Reads the data and separates it into training data, validation data and testing data
raw_data = reader.ptb_raw_data(data_dir)
train_data, valid_data, test_data, _ = raw_data

#reads 1 mini-batch
itera = reader.ptb_iterator(train_data, batch_size, num_steps)
first_touple=itera.next()
x=first_touple[0]
y=first_touple[1]

x.shape
x[0:3]
size = hidden_size

_input_data = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]
_targets = tf.placeholder(tf.int32, [batch_size, num_steps]) #[30#20]

#creates dictionary for placeholder with first mini-batch
feed_dict={_input_data:x, _targets:y}

session.run(_input_data,feed_dict)

#2 layer LSTM network
lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

#initialise states of said networks
_initial_state = stacked_lstm.zero_state(batch_size, tf.float32)
_initial_state

session.run(_initial_state,feed_dict)

#embedding
embedding = tf.get_variable("embedding", [vocab_size, hidden_size])  #[10000x200]
session.run(tf.global_variables_initializer())
session.run(embedding, feed_dict)

# Define where to get the data for our embeddings from
inputs = tf.nn.embedding_lookup(embedding, _input_data)  #shape=(30, 20, 200)
inputs
session.run(inputs[0], feed_dict)

#constructing recurrent neural network
outputs, new_state =  tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=_initial_state)
outputs
session.run(tf.global_variables_initializer())
session.run(outputs[0], feed_dict)
#reshape output tensor
output = tf.reshape(outputs, [-1, size])
output
session.run(output[0], feed_dict)

#logistic unit to return probability of output word

softmax_w = tf.get_variable("softmax_w", [size, vocab_size]) #[200x1000]
softmax_b = tf.get_variable("softmax_b", [vocab_size]) #[1x1000]
logits = tf.matmul(output, softmax_w) + softmax_b

session.run(tf.global_variables_initializer())
logi = session.run(logits, feed_dict)
logi.shape

First_word_output_probablity = logi[0]
First_word_output_probablity.shape

#prediction maximum probability

embedding_array= session.run(embedding, feed_dict)
np.argmax(First_word_output_probablity)

y[0][0]

_targets

#compares logit with target
targ = session.run(tf.reshape(_targets, [-1]), feed_dict)
first_word_target_code= targ[0]
first_word_target_code
first_word_target_vec = session.run( tf.nn.embedding_lookup(embedding, targ[0]))
first_word_target_vec

#objective function - minimise loss function
loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(_targets, [-1])],[tf.ones([batch_size * num_steps])])
session.run(loss, feed_dict)

cost = tf.reduce_sum(loss) / batch_size

session.run(tf.global_variables_initializer())
session.run(cost, feed_dict)
# store the new state as final state
final_state = new_state

#define optimiser using Gradient Descent Optimiser

# Create a variable for the learning rate
lr = tf.Variable(0.0, trainable=False)
# Create the gradient descent optimizer with our learning rate
optimizer = tf.train.GradientDescentOptimizer(lr)

#trainable variable
# Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
tvars = tf.trainable_variables()
tvars
tvars=tvars[3:]
[v.name for v in tvars]

#calculate fradients based on the loss function
cost
tvars

#gradient
var_x = tf.placeholder(tf.float32)
var_y = tf.placeholder(tf.float32)
func_test = 2.0*var_x*var_x + 3.0*var_x*var_y
session.run(tf.global_variables_initializer())
feed={var_x:1.0,var_y:2.0}
session.run(func_test, feed)

#x gradient
var_grad = tf.gradients(func_test, [var_x])
session.run(var_grad,feed)
#y gradient
var_grad = tf.gradients(func_test, [var_y])
session.run(var_grad,feed)

tf.gradients(cost, tvars)
grad_t_list = tf.gradients(cost, tvars)
#sess.run(grad_t_list,feed_dict)

max_grad_norm
# Define the gradient clipping threshold
grads, _ = tf.clip_by_global_norm(grad_t_list, max_grad_norm)
grads
session.run(grads,feed_dict)

#apply opimiser to variables/gradients tuple
# Create the training TensorFlow Operation through our optimizer
train_op = optimizer.apply_gradients(zip(grads, tvars))
session.run(tf.global_variables_initializer())
session.run(train_op,feed_dict)

#create class

class PTBModel(object):

    def __init__(self, is_training):
        ######################################
        # Setting parameters for ease of use #
        ######################################
        self.batch_size = batch_size
        self.num_steps = num_steps
        size = hidden_size
        self.vocab_size = vocab_size

        ###############################################################################
        # Creating placeholders for our input data and expected outputs (target data) #
        ###############################################################################
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])  # [30#20]
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])  # [30#20]

        ##########################################################################
        # Creating the LSTM cell structure and connect it with the RNN structure #
        ##########################################################################
        # Create the LSTM unit. 
        # This creates only the structure for the LSTM and has to be associated with a RNN unit still.
        # The argument n_hidden(size=200) of BasicLSTMCell is size of hidden layer, that is, the number of hidden units of the LSTM (inside A).
        # Size is the same as the size of our hidden layer, and no bias is added to the Forget Gate. 
        # LSTM cell processes one word at a time and computes probabilities of the possible continuations of the sentence.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0)

        # Unless you changed keep_prob, this won't actually execute -- this is a dropout wrapper for our LSTM unit
        # This is an optimization of the LSTM output, but is not needed at all
        if is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        # By taking in the LSTM cells as parameters, the MultiRNNCell function junctions the LSTM units to the RNN units.
        # RNN cell composed sequentially of multiple simple cells.
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)

        # Define the initial state, i.e., the model state for the very first data point
        # It initialize the state of the LSTM memory. The memory state of the network is initialized with a vector of zeros and gets updated after reading each word.
        self._initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        ####################################################################
        # Creating the word embeddings and pointing them to the input data #
        ####################################################################
        with tf.device("/cpu:0"):
            # Create the embeddings for our input data. Size is hidden size.
            embedding = tf.get_variable("embedding", [vocab_size, size])  # [10000x200]
            # Define where to get the data for our embeddings from
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # Unless you changed keep_prob, this won't actually execute -- this is a dropout addition for our inputs
        # This is an optimization of the input processing and is not needed at all
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        ############################################
        # Creating the input structure for our RNN #
        ############################################
        # Input structure is 20x[30x200]
        # Considering each word is represended by a 200 dimentional vector, and we have 30 batchs, we create 30 word-vectors of size [30xx2000]
        # inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, num_steps, inputs)]
        # The input structure is fed from the embeddings, which are filled in by the input data
        # Feeding a batch of b sentences to a RNN:
        # In step 1,  first word of each of the b sentences (in a batch) is input in parallel.  
        # In step 2,  second word of each of the b sentences is input in parallel. 
        # The parallelism is only for efficiency.  
        # Each sentence in a batch is handled in parallel, but the network sees one word of a sentence at a time and does the computations accordingly. 
        # All the computations involving the words of all sentences in a batch at a given time step are done in parallel. 

        ####################################################################################################
        # Instanciating our RNN model and retrieving the structure for returning the outputs and the state #
        ####################################################################################################

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, inputs, initial_state=self._initial_state)

        #########################################################################
        # Creating a logistic unit to return the probability of the output word #
        #########################################################################
        output = tf.reshape(outputs, [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])  # [200x1000]
        softmax_b = tf.get_variable("softmax_b", [vocab_size])  # [1x1000]
        logits = tf.matmul(output, softmax_w) + softmax_b

        #########################################################################
        # Defining the loss and cost functions for the model's learning to work #
        #########################################################################
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])],
                                                                  [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size

        # Store the final state
        self._final_state = state

        # Everything after this point is relevant only for training
        if not is_training:
            return

        #################################################
        # Creating the Training Operation for our Model #
        #################################################
        # Create a variable for the learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Get all TensorFlow variables marked as "trainable" (i.e. all of them except _lr, which we just created)
        tvars = tf.trainable_variables()
        # Define the gradient clipping threshold
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        # Create the gradient descent optimizer with our learning rate
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # Create the training TensorFlow Operation through our optimizer
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    # Helper functions for our LSTM RNN class

    # Assign the learning rate for this model
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    # Returns the input data for this model at a point in time
    @property
    def input_data(self):
        return self._input_data

    # Returns the targets for this model at a point in time
    @property
    def targets(self):
        return self._targets

    # Returns the initial state for this model
    @property
    def initial_state(self):
        return self._initial_state

    # Returns the defined Cost
    @property
    def cost(self):
        return self._cost

    # Returns the final state for this model
    @property
    def final_state(self):
        return self._final_state

    # Returns the current learning rate for this model
    @property
    def lr(self):
        return self._lr

    # Returns the training operation defined for this model
    @property
    def train_op(self):
        return self._train_op

"""

##APPLY CHAR MODEL for TEXT GENERATION ##

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)


    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

#parameters: batch, number_of_bach, batch_size and seq_length
seq_length = 50 # RNN sequence length
batch_size = 60  # minibatch size, i.e. size of data in each epoch
num_epochs = 125 # you should change it to 50 if you want to see a relatively good results
learning_rate = 0.002
decay_rate = 0.97
rnn_size = 128 # size of RNN hidden state (output dimension)
num_layers = 2 #number of layers in the RNN

data = "https://ibm.box.com/shared/static/a3f9e9mbpup09toq35ut7ke3l3lf03hg.txt"
with open('input.txt', 'r') as f:
    read_data = f.read()
    print (read_data[0:100])
f.closed

data_loader = TextLoader('', batch_size, seq_length)
vocab_size = data_loader.vocab_size
print ("vocabulary size:" ,data_loader.vocab_size)
print ("Characters:" ,data_loader.chars)
print ("vocab number of 'F':",data_loader.vocab['F'])
print ("Character sequences (first batch):", data_loader.x_batches[0])

#input/output
x,y = data_loader.next_batch()
x
x.shape  #batch_size =60, seq_length=50
y
# Defining stacked RNN Cell
cell = tf.contrib.rnn.BasicRNNCell(rnn_size)#
# a two layer cell
stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
# hidden state size
stacked_cell.output_size

#tracks state variable keeps output and new_states of the LSTM
stacked_cell.state_size

#defines input data
input_data = tf.placeholder(tf.int32, [batch_size, seq_length])# a 60x50
input_data

#defines target data
targets = tf.placeholder(tf.int32, [batch_size, seq_length]) # a 60x50
targets

#returns zero-filled state  tensor
initial_state = stacked_cell.zero_state(batch_size, tf.float32) #why batch_size ? 60x128

#check input data
session = tf.Session()
feed_dict={input_data:x, targets:y}
session.run(input_data, feed_dict)

#embedding
with tf.variable_scope('rnnlm', reuse=False):
    softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])  # 128x65
    softmax_b = tf.get_variable("softmax_b", [vocab_size])  # 1x65)
    # with tf.device("/cpu:0"):

    # embedding variable is initialized randomely
    embedding = tf.get_variable("embedding", [vocab_size, rnn_size])  # 65x128

    # embedding_lookup goes to each row of input_data, and for each character in the row, finds the correspond vector in embedding
    # it creates a 60*50*[1*128] matrix
    # so, the first elemnt of em, is a matrix of 50x128, which each row of it is vector representing that character
    em = tf.nn.embedding_lookup(embedding, input_data)  # em is 60x50x[1*128]
    # split: Splits a tensor into sub tensors.
    # syntax:  tf.split(split_dim, num_split, value, name='split')
    # it will split the 60x50x[1x128] matrix into 50 matrix of 60x[1*128]
    inputs = tf.split(em, seq_length, 1)
    # It will convert the list to 50 matrix of [60x128]
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

#embed variable is initialised with random values
    session.run(tf.global_variables_initializer())
    # print embedding.shape
    session.run(embedding)

# first elemnt of em, is a matrix of 50x128, which each row of it is vector representing that character
#em = tf.nn.embedding_lookup(embedding, input_data)
emp = session.run(em,feed_dict={input_data:x})
print (emp.shape)
emp[0]

#60 seconds
inputs = tf.split(em, seq_length, 1)
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
inputs[0:5]

#feeding batch of 50 sequence to RNN

session.run(inputs[0],feed_dict={input_data:x})

#outputs is 50x[60*128]
outputs, new_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, initial_state, stacked_cell, loop_function=None, scope='rnnlm')
new_state

#testing batch
outputs[0:5]

first_output = outputs[0]
session.run(tf.global_variables_initializer())
session.run(first_output,feed_dict={input_data:x})

#reshape data back to original array
output = tf.reshape(tf.concat( outputs,1), [-1, rnn_size])
output

logits = tf.matmul(output, softmax_w) + softmax_b
logits

probs = tf.nn.softmax(logits)
probs

session.run(tf.global_variables_initializer())
session.run(probs,feed_dict={input_data:x})


grad_clip =5.
tvars = tf.trainable_variables()
tvars

"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(num_epochs): # num_epochs is 5 for test, but should be higher
        sess.run(tf.assign(model.lr, learning_rate * (decay_rate ** e)))
        data_loader.reset_batch_pointer()
        state = sess.run(model.initial_state) # (2x[60x128])
        for b in range(data_loader.num_batches): #for each batch
            start = time.time()
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y, model.initial_state:state}
            train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
            end = time.time()
        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                .format(e * data_loader.num_batches + b, num_epochs * data_loader.num_batches, e, train_loss, end - start))
        with tf.variable_scope("rnn", reuse=True):
            sample_model = LSTMModel(sample=True)
            print sample_model.sample(sess, data_loader.chars , data_loader.vocab, num=50, prime='The ', sampling_type=1)
            print '----------------------------------'

"""