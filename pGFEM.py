from gauss import grule
import sys, os, json, inspect
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from shape_fns_p import legendre, shapes
from shape_fns import fns_l2
from bodyforce import BodyF
from elastic import constitutive

"""
Questions: 
    1, dN for the enrichment function ?
    2, sub K should be calculated seperatedly ?
    3, how do I calculate the u (displacement vector) in the end?
    
"""

p=1
h = 0.5
nodes = np.array([[0.0],[0.5],[1.0]])
elems = np.array([[0, 1],[1, 2]])
bcs = [[0, 2], [0.0, 0.0]]
load = [[0, 2], [0, 0]]
nnodes, dpn_0 = nodes.shape;   # node count and dofs per node
dpn_er = 1;                    # enriched dof per node, so one node one enrichment
dpn = dpn_0 + dpn_er;          # total dof per node
dofs = dpn * nnodes            # total number of dofs 
material = tuple([1, 1])
gauss = grule(2)

K = sp.lil_matrix((dofs,dofs))
F = np.zeros(dofs)

#-- Assembling stiffness matrix and force vector...
for e, conn in enumerate(elems):
      # coordinate array for the element
      X = nodes[conn]
      ldofs_0 = dpn_0 * len(conn)
      ldofs_er = dpn_er * len(conn)
      k_0 = np.zeros((ldofs_0, ldofs_0))
      k_er = np.zeros((ldofs_er, ldofs_er))
      k_0_er = np.zeros((ldofs_0, ldofs_er))
      k_er_0 = np.zeros((ldofs_er, ldofs_0))
      f_0 = np.zeros(ldofs_0)
      f_er = np.zeros(ldofs_er)

      # element degree of freedom
      eft_0 = np.array([dpn_0 * n + i for n in conn for i in range(dpn_0)])
      eft_er = np.array([dpn_er * n + i + nnodes for n in conn for i in range(dpn_er)])
      eft = np.append(eft_0, eft_er)
      print('element dof:',eft)

      # derive element k matrix
      for i, xi in enumerate(gauss.xi):
          N_0, dN_0 = fns_l2(xi, X)
          x = np.dot(X.T,[N_0[0],N_0[1]])
          L = ((x-X)/h)**p   
          N_er = np.ones(np.size(L)); dN_er = np.ones(np.size(L))        
          for i in range(np.size(L)):
              for j in range(np.size(L)):
                  if i==j: 
                      N_er[i] = N_0[i] * L[i]
                      #dN_er[i] = dN_0[i] * L[i] + N_0[i] * (p * (x-X[i])**(p-1)*((X[1]-X[0])/2))/ (h**p)
                      dN_er[i] = dN_0[i] * L[i] + N_0[i] * (p * (((x-X[i])/h)**(p-1))*((X[1]-X[0])/2*h))
          j = h/2; Jinv = 1/j
          
          B_0 = np.dot(dN_0,Jinv)
          BB_0 = np.kron(B_0.T,np.identity(dpn_0)) 
          
          B_er = np.dot(dN_er,Jinv)
          BB_er = np.kron(B_er.T,np.identity(dpn_er)) 
          
          matDT = 1
          k_0  += gauss.wgt[i] * j * np.dot(np.dot(BB_0.T, matDT), BB_0)  
          k_er += gauss.wgt[i] * j * np.dot(np.dot(BB_er.T, matDT), BB_er)
          k_0_er += gauss.wgt[i] * j * np.dot(np.dot(BB_0.T, matDT), BB_er)
          k_er_0 += gauss.wgt[i] * j * np.dot(np.dot(BB_er.T, matDT), BB_0)

      # assemble global K matrix     
      K[eft_0[:,np.newaxis], eft_0] += k_0
      K[eft_er[:,np.newaxis], eft_er] += k_er
      K[eft_0[:,np.newaxis], eft_er] += k_0_er
      K[eft_er[:,np.newaxis], eft_0] += k_er_0
      print(K.toarray())

      # derive body force vector f
      bodyf = grule(p+3)
      for i, xi in enumerate(bodyf.xi):
          N_0, dN_0 = fns_l2(xi,X)
          x = np.dot(X.T,[N_0[0],N_0[1]])
          j = h/2
          matDT = 1
          f_0 += bodyf.wgt[i] * j * N_0 * BodyF(matDT, x, 1)
          f_er += bodyf.wgt[i] * j * N_er * BodyF(matDT, x, 1)

      # assemble global body force vector
      F[eft_0] += f_0
      F[eft_er] += f_er
      print('F:',F)

K_0 = K[0:nnodes, 0:nnodes]
K_er = K[nnodes:, nnodes:] 
K_0_er = K[0:nnodes, nnodes:]
K_er_0 = K[nnodes:, 0:nnodes]
print(K_0.toarray())
print(K_er.toarray())
print(K_0_er.toarray())
print(K_er_0.toarray())
F_0 = F[0:nnodes]
F_er = F[nnodes:]

#-- Applying boundary conditions...
zero = bcs[0] 
F_0 -= K_0[:, zero] * bcs[1]  
K_0[:, zero] = 0;
K_0[zero, :] = 0;  
K_0[zero, zero] = 1 
F_0[zero] = bcs[1]  

   # apply loads
F_0[load[0]] += load[1]


#####

   #-- Solving system of equations...
u_0 = spsolve(K_0.tocsr(), F_0)
S_0_er = spsolve(K_0.tocsr(),K_0_er)
K_er_hat = K_er - K_er_0 * S_0_er
F_er_hat = F_er - K_er_0 * u_0
u_er = spsolve(K_er_hat.tocsr(), F_er_hat)
u_00 = u_0 - S_0_er * u_er
u = np.append(u_00, u_er)
K_1 = np.hstack((K_0.toarray(), K_0_er.toarray()))
K_2 = np.hstack((K_er_0.toarray(),K_er.toarray()))
K = np.vstack((K_1, K_2))

#u = spsolve(K.tocsr(), F)

  #-- Calculating strain energy...
U = 0.5 * np.dot(np.dot(u.T, K), u)
