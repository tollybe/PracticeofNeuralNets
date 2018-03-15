"""
http://neuralnetworksanddeeplearning.com/chap1.html

REDUNDANT CODE: SAVED FOR COMPARRISON:

ONLY USING unregularized stochastic gradient descent option:
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        Def SGD: helping network object learn using SGD method which implements stochastic gradient descent
            - implements by training the neural network using mini-batch stochastic gradient descent

        :training_data: list of tuples (x,y) represents training inputs and desired outputs
        :mini_batch_size: number of epochs to train for + size of mini-batches to use when sampling
        :eta: learning rate

        if external :test_data: is provided:
            1. network will be evaluated against test data after each epoch
            2. partial progress printed out
             else:
                1. prints out epoch as complete
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] # creates a lambda function of training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]  # where k is in range of 0- the training data klist, epoch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def minibatch:

    -- SAME -- UNTIL:
    self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] # weights - (learning curve/length of sample(x,y)) * nabla_w
    self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] # bias - (learningcurve/length of sample(x,y)) * nabla_b


    def evaluate(self, test_data):
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.


        test_results =[(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)



    def cost_derivative(self, output_activations, y):
    Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations.

        return output_activations - y
"""
