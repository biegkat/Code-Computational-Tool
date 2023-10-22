"""This module solves a second order differntial equation to calculate the steady state
microtubule (MT) lenght disribution given 'pytensor.tensor.var.TensorVariable' parameters.
"""
import numpy as np
import pytensor.tensor as pt

NL = 100
LMAX = 1.0
delta = LMAX/(NL-1)

### Define first derivative (pytensor) ###
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
dl1 = pt.as_tensor(dl1)

### Define second derivative (pytensor) ###
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
dl2 = pt.as_tensor(dl2)

### Define l vector (numpy) ###
ll = np.linspace(0,LMAX,NL)

### Define identiry matrix (pytensor) ###
iden = pt.identity_like(dl1)

### Define l matrix (pytensor) ###
l = pt.as_tensor(ll)*iden

### Define matrix with first row = 1 and 0 else (pytensor) ###
row0 = pt.as_tensor(
    np.array([[(i==0)*1 for j in range(NL)]
              for i in range(NL)]))

### Define matrix with last row = 1 and 0 else (pytensor) ###
rown = pt.as_tensor(
    np.array([[(i==NL-1)*1 for j in range(NL)]
              for i in range(NL)]))

### Define matrix with first row = l and 0 else (pytensor) ###
lm1 = pt.as_tensor(
    np.array([[(i==NL-1)*ll[j] for j in range(NL)]
              for i in range(NL)]))

### Define matrix with top-left corner element = 1 and 0 else (pytensor) ###
e11 = pt.as_tensor(
    np.array([[(i==0 and j==0)*1 for j in range(NL)]
              for i in range(NL)]))

### Define matrix with bottom-left corner element = 1 and 0 else (pytensor) ###
en1 = pt.as_tensor(
    np.array([[(i==(NL-1) and j==0)*1 for j in range(NL)]
              for i in range(NL)]))

def pdist(r_max, k_max, alpha):
    """Calculates the MT length distribution.

	Args:
		- r_max (pytensor.tensor.var.TensorVariable): Non dimensional turnover rate.
		- k_max (pytensor.tensor.var.TensorVariable): Non dimensional severing rate.
		- alpha (pytensor.tensor.var.TensorVariable): Stability parameter.

	Returns:
		- out (pytensor.tensor.var.TensorVariable): Probability distribution.
    """
    
    ### Steady state Eq (3.4) ###
    lhs = dl2 + r_max*dl1 + k_max*l.dot(dl1) + (2+alpha)*k_max*iden
    
    ### Boundry conditions Eq (3.5) and Eq (3.6) ###
    lhs = lhs + row0*(dl1-lhs) + rown*(alpha*k_max*delta*lm1-lhs) + e11*r_max + en1
    rhs = pt.zeros(NL) + e11.dot(k_max*(1+alpha)*pt.ones(NL)) + en1.dot(r_max*pt.ones(NL))
    
    ### Solves lhs * out(li) = rhs ###
    out = pt.linalg.solve(lhs, rhs)
    return out

########################################## EOF ##########################################