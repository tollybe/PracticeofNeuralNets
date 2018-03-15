"""
Creating a network based on the neural networks + deep learning
to use image recognition and determine handwritten numbers between 1-9

On Python Console:

reload(Main)

#generating a network with 30 hidden neurons
network = Main.Network([784,30,10])

#taking mnist.training_data over 30 epochs, with a mini batch size of 10, and a learning rate of 3.0
network.SGD(training_data, 30, 10, 3.0, test_data=test_data)
"""


import random
import numpy as np
import json
import sys
import sklearn
import scipy


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        defines how well output compares against desired output
        :param a: output
        :param y: desired output
        :returns: cost associated with a and y
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        :param a: output
        :param y: desired output
        :returns: cost associated with an  `a` and `y``

        Note: The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)


class Network(object):
    """
    def __init__: initialises the network class including
    #num_layers = layers in order of input, hidden and output
    #sizes = number of neurons
    #biases = bias using gaussian distribution with mean 0, std = 1
              enables stochastic gradient descent for y variable, skipping input [0] layer
    #weight = weights/edges using gaussian distribution with mean 0, std = 1
              stored in np matrix vectices as x,y, skipping input and output layer
    """

    def __init__(self, sizes, cost = CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # moved to large_weight
        # self.weights = [np.random.randn(y, x) for x, y in
                        # zip(sizes[:-1], sizes[1:])]  # moved to large_weight
        self.default_weight_initialiser()
        self.cost = cost

    def default_weight_initialiser(self):
        """
        Initialises:
        :Bias: sets gaussian distribution with mean 0, std = 1
        :Weight: sets gaussian distribution with mean 0,std = 1 over the sqr(number of weights connecting to same neuron)
        """
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initialiser(self):
        """
        Initialises:
        :Bias: sets gaussian distribution with mean 0, std = 1
        :Weight: sets gaussian distribution with mean 0, std = 1
        ** Used as comparrison for default
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        def feedforward: adding a feed forward method which input :a:, returns the corresponding output of
        :a: input
        :w: weights
        :b: bias
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    #SGD BACKPROPOGATION WOULD BE INSERTED HERE

    def SGD(self, training_data, epochs, mini_batch_size, eta,lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        Def SGD: helping network object learn using SGD method which implements stochastic gradient descent
            - implements by training the neural network using mini-batch stochastic gradient descent

        :training_data: list of tuples (x,y) represents training inputs and desired outputs
        :mini_batch_size: number of epochs to train for + size of mini-batches to use when sampling
        :eta: learning rate
        :lmbda: regularisation parameter
        :evaluation_data: training data that unseen/unused and is used to evaluate data
        :monitor_evaluation_cost:
        :monitor_evaluation_accuracy:
        :monitor_training_cost:
        :monitor_training_accuracy:
        :return: evaluation accuracy + cost, training accuracy + cost
        """
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            print ("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print()
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy


    def update_mini_batch(self, mini_batch, eta,lmbda, n):
        """
        Def update_mini_batch: Update the network's weights and biases by applying gradient descent using backpropagation to all training examples from mini batch.
        :mini_batch: list of tuples "(x,y)"
        :eta: learning rate
        :lmda: (ADDED) regularisation parameter
        :n: (ADDED) total size of training set
        :b: bias
        :nb: nabla_b
        :dnb: change/update in nabla_b
        :w: weight
        :dnw: change/update in nabla_w
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases] # creates lambda np.zeros(b.shape) using b
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # creates lambda of np.zeros(w.shape) using w
        for x, y in mini_batch:
            # invokes backpropogation algorithm to compute gradient of cost function
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]  # weight decay -> (1- learning curve * (regularisation parameter/total training set )) * (weights - (learning curve/length of sample(x,y)) * nabla_w)
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)] # bias - (learningcurve/length of sample(x,y)) * nabla_b

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1],(activations[-1], y))
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """
        :data: training data
        :convert: If False, data = training data, if True, data = validation or test data
        :returns: the correct number of training data inputs from neuron'
        output is assumed to be the index of whichever neuron in thes final layer when compared to evaluation
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        :data: training data
        :lmbda: regularisation parameter
        :convert: If False, data = training data, if True, data = validation or test data
        :returns: Return the total cost for the data set ``data`
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


##Misc. functions
def vectorized_result(j):
    """
    def: converts a digit (0..9) into corresponding desired output from neural network
    :j: desired output assigned 1.
    :returns: 10D unit vector with a 1.0 in the jth position and zeroes elsewhere
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """
    Defining signmoid function as the parameter:z: using the equation : http://neuralnetworksanddeeplearning.com/chap1.html#eqtn4
    :z: is output as a vector/numpy array
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


