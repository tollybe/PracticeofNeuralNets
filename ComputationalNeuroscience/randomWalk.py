"""Random walk code according to the neuroscience book"""
import random as random
import matplotlib.pyplot as plt

mu = 0.25 #mean variance
sigma = 0.25 #std
thr = 10 #threshold for value to reach
trialNumber = 1000 #range of attempts
rts =[] #reaction time array
for i in range(trialNumber + 1):
    start = 0
    timer = 0
    while (start < 0):
        myrandnum = random.normalvariate(mu,sigma)
        timer = timer + 1
    rts.append(timer)
plt.hist(rts,20)
plt.show()


"""
building a Box-Muller Transformation for a random walk

"""

import matplotlib.pyplot as plt
import numpy as np

def gaussian(u1,u2):
    """
    defining the box muller transformation method for generating uniform random numbers
    :param u1: independent variable between 0-1
    :param u2: independent variable between 01
    :return: returns z1, z2
    """
    z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
    z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
    return z1,z2
    
# uniformly distributed values between 0 and 1
u1 = np.random.rand(1000)
u2 = np.random.random(1000)

# run the transformation
z1,z2 = gaussian(u1,u2)

# plotting the values before and after the transformation
plt.figure()

plt.subplot(221) #the first row of graphs
plt.hist(u1) #histogram of u1
plt.subplot(222)
plt.hist(u2) #histogram of u2

plt.subplot(223)
plt.hist(z1) #histogram of z1
plt.subplot(224)
plt.hist(z2) #histogram of z2

plt.show()
