"""
A dynamical systems approach to the study of the large-scale behavior of neuronal population.
Their approach doesn't mind the behavior of single neurons, but works at the level of population firing rates
(roughly, the number of neurons that fire in the unit time) for two subpopulation
: the inhibitory neurons and the excitatory neurons
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

# Parameter definitions
def sigmoid(x, a, thr):
    return 1 / (1 + np.exp(-a * (x - thr)))

# For a stable limit cycle and a stable fixed point
# couplings
c1 = 16 #the strength of the exitatory to exitatory
c2 = 12 #inhibitory to excitatory
c3 = 15 #exitatory to inhibitory
c4 = 3 #inhibitory to inhibitory

# refractory periods
rE = 1
rI = 1

# external inputs
P = 1.
Q = 1

# nonlinear functions
def Se(x): #the input-frequency characteristic of the excitatory neurons
    aE = 1.3 #threshold
    thrE = 4 #alpha
    return sigmoid(x, thrE, aE) - sigmoid(0, thrE, aE)

def Si(x): #the inhibitory ones
    aI = 2
    thrI = 3.7
    return sigmoid(x, thrI, aI) - sigmoid(0, thrI, aI)

# this function returns the right hand side of the Wilson-Cowan equation (both, in a 2-vector)
def WilsonCowan(y, t):
    E = y[0]
    I = y[1]
    y1 = -E + (1 - rE * E) * Se(c1 * E - c2 * I + P)
    y2 = -I + (1 - rI * I) * Si(c3 * E - c4 * I + Q)
    return [y1, y2]

# minimum and maximum E and I values we want displayed in the graph
minval = -.1
maxval = .6
resolution = 50
# State variables
x1 = np.linspace(minval, maxval, resolution)
x2 = np.linspace(minval, maxval, resolution)
# Create a grid for evaluation of the vector field
x1, x2 = np.meshgrid(x1, x2)
# Evaluate the slopes
X1, X2 = WilsonCowan([x1, x2], 0)
# Compute the magnitude vector
M = np.hypot(X1, X2)
# Normalize the slopes vectors (for the field plot)
X1, X2 = X1/M, X2/M

# Solving and plotting
fixed_p = []
y1 = x1.ravel()
y2 = x2.ravel()
for i in range(resolution**2):
    # find a zero
    sol, infodict, ier, mesg = fsolve(WilsonCowan, [y1[i], y2[i]], args=(0), full_output=1)
    if ier == 1: # I exclude the cases where fsolve didn't converge
        fixed_p.append(sol)

fixed_p = np.array(fixed_p).T

# Numerical ODE integration
# simulation duration and step size
time = np.linspace(0, 100, 2000)

# starting point, hopefully inside the basin of attraction of our attractor
E0, I0 = 0.39, 0.49 # try changing this

# find the solution with scint.odeint
odesol = odeint(WilsonCowan, [E0, I0], time)
# separate the two solutions
exc_timeseries, inh_timeseries = odesol.T

# Plotting
# plotting the vector field in the state space (E, I)
plt.figure(figsize=(10, 10))
plt.quiver(x2, x1, X2, X1, color = 'slateblue', alpha=.5) #to show electrical potential
plt.xlim([minval, maxval])
plt.ylim([minval, maxval])
plt.xlabel(r'$I$', fontsize=13) # yes, you can use Latex code!
plt.ylabel(r'$E$', fontsize=13)
plt.grid()

# plot the solution in the state space
plt.plot(inh_timeseries, exc_timeseries, '.-')

# plot the starting point
plt.scatter(I0, E0, marker='*', s=300, label="Starting point")
plt.legend(loc="upper left", fontsize = 13)

# plot the fixed points we identified
plt.scatter(fixed_p[1], fixed_p[0], marker='o', s=50, label="Stationary points")
plt.legend(loc="upper left", fontsize = 13)

# plot the solution in time
plt.figure(figsize=(10,5))
plt.ylabel(r'$E, I$', fontsize = 13)
plt.xlabel(r'$t$', fontsize = 13)
plt.plot(time, exc_timeseries, '.-', color = 'pink', label="excitatory")
plt.plot(time, inh_timeseries, '.-', color = 'skyblue', label="inhibitory")
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.legend(loc = "upper right", fontsize = 11)
plt.show()

def ydot(t, y, flag, I_ext):
    pass

def wilson_ode(g, E, tau, x0, y, I_ext):
    # K = 1, Ca = 2, KCa = 3, Na = 4
    g[1] = 26
    g[2] = 2.25
    g[3] = 9.5
    g[4] = 1
    g = g.T
    E[1] = -.95
    E[2] = 1.2
    E[3] = E[1]
    E[4] = .5
    E = E.T
    tau[1] = 1 / 4.2
    tau[2] = 1 / 14
    tau[3] = 1 / 45
    tau = tau.T

    V = y[4]
    y3 = y[1 : 4]

    x0[1] = 1.24 + 3.7 * V + 3.2 * V**2
    x0[2] = 4.205 + 11.6 * V + 8 * V**2
    x0[3] = 3 * y[2]
    x0 = x0.T

    ydot = np.multiply(tau, (x0 - y3))

    y3[4] = 17.8 + 47.6 * V + 33.8 * V**2

    I1 = np.multiply(g, y3)
    I = np.multiply(I1, (y[4] - E))

    ydot[4] = I_ext - sum(I)

def wilson_integrate():
    y0 = np.zeros(1, 4)
    y0[4] = -1
    I_ext = 0
    tspan = [0, 100]

    odesol2 = odeint(WilsonCowan, y0, tspan, [], I_ext)

    [t, y] = odesol2

    plt.plot(t, 100 * y[:, 4])
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential')
    plt.show()