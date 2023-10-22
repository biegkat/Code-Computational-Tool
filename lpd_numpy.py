"""This module solves a second order differntial equation to calculate the steady state
microtubule (MT) lenght disribution given 'float' parameters.
"""
import numpy as np
from scipy.linalg import solve

NL = 100            #Number of grid points
LMAX = 1            #LMAX=1 -> using scaled parameters 

### Define grid ###
l = np.linspace(0,LMAX,NL)
delta = LMAX/(NL-1)

### Define first derivative ###
dl1 = np.zeros((NL,NL))
for i in range(1,NL-1):
    dl1[i,i-1] = -1
    dl1[i,i+1] = 1
dl1[0,0] = -3
dl1[0,1] = 4
dl1[0,2] = -1
dl1[-1,-1] = 3
dl1[-1,-2] = -4
dl1[-1,-3] = 1
dl1 = dl1/2/delta

### Define second derivative ###
dl2 = np.zeros((NL,NL))
for i in range(1,NL-1):
    dl2[i,i-1] = 1
    dl2[i,i+1] = 1
    dl2[i,i] = -2
dl2[0,0] = 2
dl2[0,1] = -5
dl2[0,2] = 4
dl2[0,3] = -1
dl2[-1,-1] = 2
dl2[-1,-2] =-5
dl2[-1,-3] = 4
dl2[-1,-4] = -1
dl2 = dl2/delta**2

### Define identity operator ###
iden = np.eye(NL)

def pdist(r, k, alpha):
    """Calculates the MT length distribution.

    Args:
    	- r_max (float): Non dimensional turnover rate.
    	- k_max (float): Non dimensional severing rate.
    	- alpha (float): Stability parameter.

    Returns:
        - out (numpy.ndarray of float): Probability distribution.
    """

    ### Steady state Eq. (3.4) ###
    lhs = dl2 + (r+k*l[:,None])*dl1 + (2+alpha)*k*iden
    rhs = np.zeros(NL)

    ### Boundry condition Eq. (3.6) ###
    lhs[0,:] = dl1[0,:]
    lhs[0,0] = lhs[0,0]+r
    rhs[0] = k*(1+alpha)

    ### Boundry condition Eq. (3.8) ###
    lhs[-1,:] = k*alpha*l*delta
    lhs[-1,0] = 1+ k*alpha*l[0]*delta
    rhs[-1] = r

    out = solve(lhs, rhs)
    return out

########################################## EOF ##########################################