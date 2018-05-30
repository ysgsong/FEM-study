"""
Advanced Finite Methods Assignment 1
Problem 1 Question 5 p-GFEM

Created by: Yi Song
30-05-2018

"""
from gauss import grule
import sys, os, json, inspect
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


#%% Body Forces
def BodyF(C,x):
    d2u_dx2 = -(2*(0.5+0.5**3*0.8*(x-0.2)))/((0.25*(x-0.2)**2+1)**2) 
    f = - C*d2u_dx2
    
    return f

#%% Shape Functions
def shapefunc(xi, X, p, h):
    # Compute N
    phi = 0.5*np.array([1-xi, 1+xi])
    x = np.dot(X.T, phi)
    L1 = np.array([]); L2 = np.array([])
    for i in range(p):
        L1 = np.append(L1,((x-X[0])/h)**i)
        L2 = np.append(L2,((x-X[1])/h)**i)
        print(L1)
        print(L2)
    M1 = np.array([L1, L2])
    M2 = np.array([[1, 0]])
    LL = np.kron(M1, M2)
    LL[1] = np.roll(LL[1],1)
    print(LL)
    N = np.dot(phi, LL)
    
    # Compute B
    dphi = np.array([[-1/2],[1/2]])  
    B = 2/h * dphi
    b = np.array([])
    if p >=2:
        for n in range(p-1):
            i = n + 1
            for j in range(2):
                b_0 = 2/h * dphi[j] * ((x-X[j])/h)**i + phi[j]*((x-X[j])/h)**i*(1/(x-X[j]))
                b = np.append(b,b_0)
        B = np.append(B,b)
    return N, B, phi        

#%% Solving K matrix singular problem


#%% Main
# Initialization 
p = 2
h = 0.5
nodes = np.array([[0.0],[0.5],[1.0]])
elems = np.array([[0,1],[1,2]])
bcs = [[0, 2], [0.0, 0.0]]
load = [[0, 2], [0, 0]]
nnodes, dpn_0 = nodes.shape    # node count and dofs per node
dpn_er = 1                     # enriched dof per node, so one enrichment per node
dpn = dpn_0 + dpn_er           # total dof per node
dofs = dpn * nnodes            # total number of dofs 
C = 1                          # material C matrix

# K and F initialization
K = sp.lil_matrix((dofs,dofs))
F = np.zeros(dofs)

# Gauss integration point
gauss_k = grule(p+1)
gauss_f = grule(p+5)

# Assembling stiffness matrix and force vector
for e,conn in enumerate(elems):
      X = nodes[conn]
      ldofs = dpn * len(conn)
      k = np.zeros((ldofs, ldofs))
      f = np.zeros(ldofs)
      
      eft_0 = np.array([dpn_0 * n + i for n in conn for i in range(dpn_0)])
      eft_er = np.array([len(elems) + eft_0[1] + i for i in range(dpn)])
      eft = np.append(eft_0, eft_er)
      
      # Stiffness matrix
      for i, xi in enumerate(gauss_k.xi):
          N, B, phi = shapefunc(xi, X, p, h)
          j = h/2
          BB = np.kron(B.T,np.identity(ldofs))  
          k += gauss_k.wgt[i] * j * np.dot(np.dot(BB, C), BB.T)
      
      # assemble global K matrix
      K[eft[:, np.newaxis], eft] += k  
      
      # force vactor
      for i, xi in enumerate(gauss_f.xi):
          N, B, phi = shapefunc(xi, X, p, h)
          x = np.dot(X.T, phi)
          j = h/2
          f[0:2] += gauss_f.wgt[i] * j * N[0:2] * BodyF(C,x) 
          
      # assemble global body force vector
      F[eft] += f
 
 # Applying boundary conditions
zero = bcs[0]               # array of rows/columns which are to be zeroed out
F -= K[:, zero] * bcs[1]    # modify right hand side with prescribed values
K[:, zero] = 0;
K[zero, :] = 0;             # zero-out rows/columns
K[zero, zero] = 1           # add 1 in the diagonal
F[zero] = bcs[1]            # prescribed values

# apply loads
F[load[0]] += load[1]

# Solving system of equations
u = spsolve(K.tocsr(), F)

# Calculating strain energy
U = 0.5 * np.dot((u.T*K), u)     
      
      
      
      
      
      
      
      