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
# %% function to calculate element dof
def elementdofs(e, conn, p, dpn_0, nnodes):
    #eft = np.array([dpn_0 * n + i for n in conn for i in range(dpn_0)]) 
    #if e == 0:
    #   if p > 1:
    #       eft = np.append(eft, [2+eft[-1]+i for i in range(4)])
    #   if p > 2:
    #       eft = np.append(eft, [eft[-1] + 3 + i for i in range(4)])
    #if e > 0:
    #   for i in range(p-1):
    #       eft = np.append(eft, [eft[-1] + 3 + i for i in range(4)])

    eft_0 = np.array([dpn_0 * n + i for n in conn for i in range(dpn_0)]) 
    if p == 1:
        eft = eft_0
    elif p > 1:    
         if e == 0:
            a1 = np.array([eft_0[1] + 2 + i for i in range(2)])
            b1 = np.array([eft_0[1] + 4 + i for i in range(2)])
         if e == 1:
            a1 = np.array([eft_0[1] + 3 + i for i in range(2)])
            b1 = np.array([eft_0[1] + 5 + i for i in range(2)])
         #if p > 2:
         for i in range (p-2):
                a2 = np.array([2*(nnodes - 1) + 1 + a1[-1] + i for i in range(2)])
                b2 = np.array([2*(nnodes - 1) + 1 + b1[-1] + i for i in range(2)])
                a1 = np.append(a1,a2)
                b1 = np.append(b1,b2)
    
         eft_1 = np.append(eft_0[0], a1)
         eft_2 = np.append(eft_0[1], b1)
         eft = np.append(eft_1, eft_2)
  
    return eft

#%% Body Forces
def BodyF(C,x):
    d2u_dx2 = -(2*(0.5+0.5**3*0.8*(x-0.2)))/((0.25*(x-0.2)**2+1)**2) 
    f = - C*d2u_dx2
    
    return f

#%% Shape Functions
def stdFEM_shapefns(xi, X):
    '''Provide the std.FEM shape functions.
       [1D problem]
       [2 nodes per element]
    '''
    N = 0.5*np.array([1 - xi, 1 + xi])
    dN = np.array([[-1/2],[1/2]])
    return N, dN

def pGFEM_enrichment_fns(x, X, h, p):
    '''Provide the enrichment functions [L1;L2] used for pGFEM.
       [1D problem]
       [2 nodes per element]
       
       where L1 is the enrichment function for the first node, L2 is for the second node.
       The size of the enrichment functions increases with the increasing of p, 
       if p = 1 (at least), no enrichment.    
    '''
    L1 = np.array([]); L2 = np.array([])
    for i in range(p):
        L1 = np.append(L1,((x-X[0])/h)**i)
        L2 = np.append(L2,((x-X[1])/h)**i)
    L = np.array([L1, L2])    
    return L


def shapefns(xi, X, p, h):
    '''Compute the total shape function based on stdFEM_shapefns and 
       pGFEM_enrichment_fns
    '''
    N_0, dN_0 = stdFEM_shapefns(xi, X)
    x = np.dot(X.T, N_0)
    L = pGFEM_enrichment_fns(x, X, h, p)
    '''for calculation reason, this L will be further transferred into a matrix 
       with only one element and a zero in each column.
    '''    
    calculation_matrix = np.array([[1, 0]])
    LL = np.kron(L,calculation_matrix)
    LL[1] = np.roll(LL[1],1)
    '''for calculation reason, 
       transfer the std.shapefns into a matrix [phi1 phi1; phi2 phi2]
    '''
    std_transf= (np.array([N_0,N_0])).T
    N_matrix = np.dot(std_transf,LL)
    N = np.append(N_matrix[0,1:], N_matrix[1,1:])

    '''Compute the derivative of the total shape function for pGFEM
    '''
    B = 2/h * dN_0
    i = 0
    b_1 = np.array([])
    for n in range(p-1):
        m = n + 1
        b = np.array([])
        for j in range(len(X)):
            b_0 = 2/h * dN_0[i] * ((x-X[j])/h)**m + m * N_0[i]*((x-X[j])/h)**m*(1/(x-X[j]))
            b = np.append(b,b_0)  #'''append: phi1*L11 phi1*L12 '''
        b_1 = np.append(b_1,b) #'''append: phi1*L11 phi1*L12 phi1*L11^2 phi1*L12^2'''
    b_2 = np.append(B[0], b_1)  #'''append: dphi1 phi1*L11 phi1*L12 phi1*L11^2 phi1*L12^2'''  
    B_1 = b_2
    
    i = 1
    b_1 = np.array([])
    for n in range(p-1):
        m = n + 1
        b = np.array([])
        for j in range(len(X)):
            b_0 = 2/h * dN_0[i] * ((x-X[j])/h)**m + m * N_0[i]*((x-X[j])/h)**m*(1/(x-X[j]))
            b = np.append(b,b_0) 
        b_1 = np.append(b_1,b)
    b_2 = np.append(B[1], b_1)
    B_2 = b_2

    B_final = np.append(B_1, B_2)
    
    return N,B_final,N_0,dN_0      

#%% Solving K matrix singular problem
def singular_solver(K,F):
    T = np.zeros((9,9))
    for i in range(9):
        for j in range(9):        
            if i == j: 
                kron = 1
                T[i,j] = kron / np.sqrt(K[i,j])
            elif i != j: 
                kron = 0
                T[i,j] = 0 
    #print('T',T)         
    K_new = (T @ K) @ T      # K of the new system
    #print(K_new)
    F_new = T @ F            # F of the new system
 
    eps   = 5*10**(-10)
    K_eps = K_new + eps*np.eye(9)
    
    u_i = np.linalg.solve(K_eps, F_new)
    print(u_i)
    while True:
          print('here')
          r_i = F_new - K_new.dot(u_i)
          e_i = np.linalg.solve(K_eps, r_i)
          u_i += e_i
 
          test_flag = np.dot((e_i.T * K_new),e_i) / np.dot((u_i.T * K_new),u_i)
          test_flag[0] = test_flag[2] = 0
          flag = np.linalg.norm(test_flag)
          print('flag',flag)
          if flag < eps:
             break

    u = np.matmul(T,u_i) 
    return u
       
#%% Main
# Initialization 
p = 2
h = 0.5
nodes = np.array([[0.0],[0.5],[1.0]])
elems = np.array([[0,1],[1,2]])
bcs = [[0, 2], [0.0, 0.0]]
load = [[0, 2], [0, 0]]
nnodes, dpn_0 = nodes.shape    # node count and dofs per node
nep = p-1                      # number of enrichment per node
dpn_er = nep * 2               # enriched dof per node
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
      
      eft = elementdofs(e, conn, p, dpn_0, nnodes)
      print(eft)
      
      # Stiffness matrix
      for i, xi in enumerate(gauss_k.xi):
          N, B, N_0, dN_0 = shapefns(xi, X, p, h)
          j = h/2
          BB = np.kron(B.T,np.identity(dpn_0))  
          k += gauss_k.wgt[i] * j * np.dot(np.dot(BB.T, C), BB)
      
      # assemble global K matrix
      K[eft[:, np.newaxis], eft] += k  
      
      # force vactor
      for i, xi in enumerate(gauss_f.xi):
          N, B, N_0, dN_0 = shapefns(xi, X, p, h)
          x = np.dot(X.T, N_0)
          j = h/2
          f += gauss_f.wgt[i] * j * N * BodyF(C,x) 
          
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
u = singular_solver(K,F)

#u = spsolve(K.tocsr(), F)

# Calculating strain energy
U = 0.5 * np.dot((u.T*K), u)     
      
      
