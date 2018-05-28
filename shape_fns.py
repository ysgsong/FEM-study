import numpy as np
from numpy.linalg import inv, det

# functions that return shape functions, their derivatives, and the global coordinate

def fns_l3(xi, X):

    N = np.array([0.5*(xi*xi - xi), 1 - xi*xi, 0.5*(xi + xi*xi)])
    dN = np.array([[0.5*(2*xi - 1)], [(-2)*xi], [0.5*(1 + 2*xi)]])
    return N, dN


def fns_l2(xi, X):

    N = 0.5*np.array([1 - xi, 1 + xi])
    dN = np.array([[-1/2],[1/2]])
    return N, dN


# compute Jacobian matrix
def jacobian(X, dN):

    # compute Jacobian matrix
    J = np.dot(np.transpose(X),dN)
    return inv(J), det(J)

