import tensorflow as tf
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
#logistic
import time
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
#activation
from mpl_toolkits.mplot3d import Axes3D

"""
plt.rcParams['figure.figsize'] = (10, 6)


##d6slope and intercept to verify the changes in the graph
a=1
b=0

X = np.arange(0.0, 5.0, 0.1) #independent variable
Y= a*X + b #dependent variable



plt.plot(X,Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

##LINEAR REGRESSION ##

#creating random poins to fit y = 3x+2 where x = x_data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data *3 + 2
#vectorising the unit point with function where Loc = position, scale = size
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0,scale= 0.1)) (y_data)

#initialise vairables a + b, with random guess

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a*x_data + b

#function that find loss value through mean[]

loss = tf.reduce_mean(tf.square(y-y_data))

#define optimiser method
optimizer = tf.train.GradientDescentOptimizer(0.5)

#train method
train = optimizer.minimize(loss)

#init variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)



train_data = []
for step in range(100):
    evals = sess.run([train,a,b])[1:]
    if step % 5 == 0:
        print(step, evals)
        train_data.append(evals)

converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(x_data, y_data, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])
plt.show()


###LOGISTIC REGRESSION - using IRIS###

#access data set to iris, splitting into training and testing sets
iris = load_iris()
iris_X, iris_y = iris.data[:-1,:], iris.target[:-1]
iris_y= pd.get_dummies(iris_y).values
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state=42)

# numFeatures is the number of features in iris input data.
#In the iris dataset, this number is '4'.
numFeatures = trainX.shape[1]

# numLabels is the number of classes iris data points can be in.
# In the iris dataset, this number is '3'.
numLabels = trainY.shape[1]

# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures]) #Iris has 4 features, so X is a tensor to hold data.
yGold = tf.placeholder(tf.float32, [None, numLabels]) #Holds correct answers matrix for 3 classes.

#assigning weights + biases

W = tf.Variable(tf.zeros([4, 3]))  # 4-dimensional input and  3 classes
b = tf.Variable(tf.zeros([3])) # 3-dimensional output [0,0,1],[0,1,0],[1,0,0]

#Randomly sample from a normal distribution with standard deviation .01

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=0.01,
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=0.01,
                                    name="bias"))


# Three-component breakdown of the Logistic Regression equation.
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")


# Number of Epochs in our training
numEpochs = 700

# Defining our learning rate iterations (decay)
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

#Defining our cost function - Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")

#Defining our Gradient Descent
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

#initialise
sess = tf.Session()
# Initialize our weights and biases variables.
init_OP = tf.global_variables_initializer()
# Initialize all tensorflow variables
sess.run(init_OP)

# argmax(activation_OP, 1) returns the label with the most probability
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
# If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Summary op for regression output
activation_summary_OP = tf.summary.histogram("output", activation_OP)
# Summary op for accuracy
accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# Summary op for cost
cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
# Merge all summaries
merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])
# Summary writer
writer = tf.summary.FileWriter("summary_logs", sess.graph)

#running program

# Initialize reporting variables
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        # Report occasional stats
        if i % 10 == 0:
            # Add epoch to epoch_values
            epoch_values.append(i)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))


#plot:
# How well do we perform on held-out test data?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP,
                                                     feed_dict={X: testX,
                                                                yGold: testY})))
#showing the plot of decline in gradient descent
plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show()

"""
###Activation Functions ##


#plots a surface (3D) for an arbitrary activation function with bias/weight between -.5 to +.5, step of .05


def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05)
    bs = np.arange(-0.5, 0.5, 0.05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=sess) \
                   for w,b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)

#start a session
sess = tf.Session();
#create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
#create a matrix of weights
w = tf.random_normal(shape=[3, 3])
#create a vector of biases
b = tf.random_normal(shape=[1, 3])
#dummy activation function
def func(x): return x
#tf.matmul will multiply the input(i) tensor and the weight(w) tensor then sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
#Evaluate the tensor to a numpy array
act.eval(session=sess)

#step function
plot_act(1.0, func)

#Sigmoid function
plot_act(1, tf.sigmoid)
act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)

#using tanH
plot_act(1, tf.tanh)
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess)

#using Relu
plot_act(1, tf.nn.relu)
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess)