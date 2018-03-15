import pylab as pyl
import math as m

voltageInitial = 0.0
change_in_time = 0.01
eSodium = 115  # Sodium  Nernst reversal potentials, in mV
eSodiumConduct = 120  # sodium maximum conductances, in mS/cm^2
ePotassium = -12  # Potassium Nernst reversal potentials, in mV
ePotassiumConduct = -12  # Potassium maximum conductances in mS/cm^2
eLeak = 10.6  # Leak Nernst reversal potentials, in mV
eLeakConduct = 0.3  # leak maximum conductances, in mS/cm^2


# updates the time frame
def upd(x, delta_x):
    return x + delta_x * change_in_time


# setting the m-at rest to = 0
def mnh0(a, b):
    return (a / (a + b))


# plotting  alpha and beta equations values for N
"""Channel gating kinetics. Functions of membrane voltage"""


def am(v):
    return (2.5 - 0.1 * v) / (m.exp(2.5 - 0.1 * v) - 1)


"""Channel gating kinetics. Functions of membrane voltage"""


def bm(v):
    return 4 * m.exp((-1) * v / 18)


# plotting for alpha and beta equation values for N
"""Channel gating kinetics. Functions of membrane voltage"""


def an(v):
    return (0.1 - 0.01 * v) / (m.exp(1 - (0.1 * v)) - 1)


"""Channel gating kinetics. Functions of membrane voltage"""


def bn(v):
    return 0.125 / m.exp((-1) * v / 80)


# plotting for alpha and beta equation values for H
"""Channel gating kinetics. Functions of membrane voltage"""


def ah(v):
    return 0.07 * m.exp((-1) * v / 20)


"""Channel gating kinetics. Functions of membrane voltage"""


def bh(v):
    return 1 / (m.exp(3 - (0.1) * v) + 1)


# setting the atrest channels( values) to 0
am0 = am(0)
bm0 = bm(0)
an0 = an(0)
bn0 = bn(0)
ah0 = ah(0)
bh0 = bh(0)

# setting the appropriate channels to their equations
m0 = mnh0(am0, bm0)
n0 = mnh0(an0, bn0)
h0 = mnh0(ah0, bh0)
"""
Equations:
Conductance:
Conductance (change in voltage/ change in time) = I+injected(time) -[conductance(sodium)m^3h(Voltage(time) -E_sodium) + conductance(potassium)n^4(Voltage(time)-E(potassium)+ conductance(leak)(Voltage(time)-Energy_leak)]

Channel opening:
m = alpha_m(Voltage)(1 -m). -beta_m(Voltage)_m.
h is D*/dt = a*(v)(1-*)-B*(v)* where * = n
"""


# initial sodium gateway
def initSodium(m, h, v):
    """
           Membrane current (in uA/cm^2)
           Sodium (Na = element name)

           |  :param V:
           |  :param m:
           |  :param h:
           |  :return:
           """
    return eSodiumConduct * (m ** 3) * h * (v - eSodium)


# initial potassium gateway
def initPotassium(m, v):
    """
          Membrane current (in uA/cm^2)
          Potassium (K = element name)

          |  :param V:
          |  :param h:
          |  :return:
          """
    return ePotassiumConduct * (m ** 4) * (v - ePotassium)


# initial leak between potassium/sodium variables
def initLeak(v):
    return eLeakConduct * (v - eLeak)

"""
selects new values from the old values and incprporates all in updates
"""
def newS(v, m, n, h, t):
    if (t < 5.0) or (t > 6.0):
        istim = 0.00
    else:
        istim = 20.00
    dv = istim - (initSodium(m, h, v) + initPotassium(n, v) + initLeak(v))
    dm = am(v) * (1 - m) - bm(v) * m
    dn = an(v) * (1 - n) - bn(v) * n
    dh = ah(v) * (1 - h) - bh(v) * h
    vp = upd(v, dv)  # update change in voltage
    tp = t + change_in_time  # update change in time values
    mp = upd(m, dm)  # update m values
    np = upd(n, dn)  # update n values
    hp = upd(h, dh)  # update h values
    return vp, mp, np, hp, tp


vs = []
ms = []
ns = []
hs = []
ts = []

a, b, c, d, e = newS(voltageInitial, m0, n0, h0, 0.0)
vs.append(a)  # adding voltage to list a
ms.append(b)  # adding m value to list b
ns.append(c)  # adding n value to list c
hs.append(d)  # adding h value to list d
ts.append(e)  # adding time value to list e
for i in range(2, 3000):
    a, b, c, d, e = newS(vs[-1], ms[-1], ns[-1], hs[-1], ts[-1])  # calling upon last variable whilst in iterative loop
    vs.append(a)  # adding voltage to list a
    ms.append(b)  # adding m value to list b
    ns.append(c)  # adding n value to list c
    hs.append(d)  # adding h value to list d
    ts.append(e)  # adding time value to list e

pyl.plot(ts, vs)
pyl.xlabel("time change")
pyl.ylabel("voltage change")
pyl.show()
