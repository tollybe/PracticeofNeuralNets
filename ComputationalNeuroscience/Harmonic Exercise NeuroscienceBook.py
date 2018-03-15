"""
import numpy, matplotlib, scipy

from matplotlib import pylab

dt = 0.05
p = -5.0
sp = 5.0

acc = [p*sp]
vel = [0.0]
s = [sp]
t = [0.0]

for i in range (1,100):
    acc.append(s[-1]*p)
    vel.append(vel[-1] + acc [-1])
    s.append(s[-1] +  vel [-1] * dt)
    t.append(dt*i)

dp = pylab.plot(t,s)
pylab.show()
"""
#matrix
import numpy as np

a = np.matrix([[1,3,-4],[2,4,6]])
b = np.matrix([[1,2,-4],[3,4,4]])

addition = a + b # result = [[2,5,-8],[5,8,10]]
c = np.matrix([[1,2,-4,-2],[2,4,6,3]])
d = np.matrix([[1,2],[3,4],[2,4],[3,-2]])
multiply = c * d

print("A+B",sum)
print("C*D",multiply)

#dot product
#
#result = np.dot(np.array(a)[:,0],b)

#sum([a[i][0]*b[i] for i in range(len(b))])