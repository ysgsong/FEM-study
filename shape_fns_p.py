import numpy as np

def legendre(xi,p):
    if p == 0:
        return 1
    elif p == 1:
        return xi
    elif p >= 2:
        return (1/p) * ((2*p-1) * xi * legendre(xi, p-1) + (1-p) * legendre(xi, p-2))
    # returns legendre polynormial of order p

def shapes(xi, p):
    global phi, dphi
    assert p > 0, "Invalid polynomial order"

    phi = [0.5*(1-xi), 0.5*(1+xi)]
    dphi = [-0.5, 0.5]

    if p >= 2:
       for j in range(2, p+1):
        phi += [1/(np.sqrt(4*j-2)) * (legendre(xi, j) - legendre(xi, j-2))]
        dphi += [np.sqrt((2*j - 1)/2) * legendre(xi, j-1)]

    return phi, dphi

