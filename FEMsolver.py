from gauss import quadrature, grule
import sys, os, json, inspect
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from shape_fns import fns_l2, fns_l3, jacobian
from bodyforce import BodyF
from elastic import constitutive
from shape_fns_p import legendre, shapes

def FEMsol(file,case):

# function to calculate the strain energy of a given mesh size

    with open(file) as f:

        #print("-- Reading file '{}'".format(file))
        data = json.load(f)  # load
        nodes = np.array(data['nodes'])  # nodes array to numpy array
        elems = np.array(data['elements'])  # elements array to numpy array
        nnodes, dpn = nodes.shape  # node count and dofs per node
        etype = data['etype']
        bcs = data['boundary']
        h = data['meshsize']
        load = data['load']
        dofs = dpn * nnodes  # total number of dofs
        material = tuple(data['material'])
        gauss = quadrature(etype, data['gauss'])  # quadrature data structure

    K = sp.lil_matrix((dofs, dofs))
    F = np.zeros(dofs)

    for e, conn in enumerate(elems):
        # coordinate array for the element
        X = nodes[conn]
        ldofs = dpn * len(conn)
        k = np.zeros((ldofs, ldofs))
        f = np.zeros(ldofs)
        eft = np.array([dpn * n + i for n in conn for i in range(dpn)])

        # derive element k matrix
        for i, xi in enumerate(gauss.xi):
            N, dN = eval('fns_{}'.format(etype))(xi, X)
            Jinv, j = jacobian(X, dN)
            B = np.dot(dN, Jinv)
            BB = np.kron(B.T, np.identity(dpn))
            matDT = constitutive(material, dpn)
            k += gauss.wgt[i] * j * np.dot(np.dot(BB.T, matDT), BB)

        # assemble global K matrix
        K[eft[:, np.newaxis], eft] += k

        # for loop: derive element body force vector
        bodyf = quadrature(etype, 5)
        for i, xi in enumerate(bodyf.xi):
            N, dN = eval('fns_{}'.format(etype))(xi, X)
            Xxi = np.dot(X.T, N)  # map xi back to X and then substitute into the bodyforce!!!!
            Jinv, j = jacobian(X, dN)
            matDT = constitutive(material, dpn)
            f += bodyf.wgt[i] * j * N * BodyF(matDT, Xxi, case)

        # assemble global body force vector
        F[eft] += f

    #print("-- Applying boundary conditions...")
    zero = bcs[0]  # array of rows/columns which are to be zeroed out
    F -= K[:, zero] * bcs[1]  # modify right hand side with prescribed values
    K[:, zero] = 0;
    K[zero, :] = 0;  # zero-out rows/columns
    K[zero, zero] = 1  # add 1 in the diagonal
    F[zero] = bcs[1]  # prescribed values

    # apply loads
    F[load[0]] += load[1]

    #print("-- Solving system of equations...")
    u = spsolve(K.tocsr(), F)

    #print("-- Calculating strain energy...")
    U = 0.5 * np.dot((u.T*K), u)

    return U, dofs, h


def pFEMsol(p,case):
  
   h = 0.5
   nodes = np.array([[0.0], [0.5], [1.0]])
   elems = np.array([[0, 1], [1, 2]])
   bcs = [[0, 2+(p-1)], [0.0, 0.0]]
   load = [[0, 2+(p-1)], [0, 0]]
   nnodes, dpn = nodes.shape                 # node count and dofs per node
   dofs = dpn * nnodes + 2*(p-1)             # total number of dofs 
   material = tuple([1, 1])
   gauss = grule(p+1)

   K = sp.lil_matrix((dofs, dofs))
   F = np.zeros(dofs)

   #-- Assembling stiffness matrix and force vector...
   for e, conn in enumerate(elems):
      # coordinate array for the element
      X = nodes[conn]
      ldofs = dpn * len(conn) + (p-1)
      k = np.zeros((ldofs, ldofs))
      f = np.zeros(ldofs)

      # element degree of freedom
      if p == 1:
          eft = np.array([dpn * n + i for n in conn for i in range(dpn)])
      if p > 1:
         if e == 0:
            eft = np.array([dpn * n for n in range(ldofs)])
         elif e > 0:
            eft_0 = np.array([1+p*(e-1), 1+p*e])
            eft_1 = np.array([1 + p*e + (n+1) for n in range(p-1)])
            eft = np.append(eft_0, eft_1)

      # derive element k matrix
      for i, xi in enumerate(gauss.xi):
          phi, dphi = shapes(xi, p)
          j = h/2; Jinv = 1/j
          B = np.dot(dphi,Jinv)
          BB = np.kron(B.T,np.identity(dpn))          
          matDT = constitutive(material, dpn)
          k += gauss.wgt[i] * j * np.dot(np.dot(BB.T, matDT), BB)

      # assemble global K matrix
      K[eft[:, np.newaxis], eft] += k

      # derive body force vector f
      bodyf = grule(p+3)
      for i, xi in enumerate(bodyf.xi):
          phi, dphi = np.array(shapes(xi,p))
          Xxi = np.dot(X.T,[phi[0],phi[1]])
          #Xxi = mapping(xi,e)
          j = h/2
          matDT = constitutive(material, dpn)
          f += bodyf.wgt[i] * j * phi * BodyF(matDT, Xxi,case)

      # assemble global body force vector
      F[eft] += f

   #-- Applying boundary conditions...
   zero = bcs[0] 
   F -= K[:, zero] * bcs[1]  
   K[:, zero] = 0;
   K[zero, :] = 0;  
   K[zero, zero] = 1 
   F[zero] = bcs[1]  

   # apply loads
   F[load[0]] += load[1]

   #-- Solving system of equations...
   u = spsolve(K.tocsr(), F)

   #-- Calculating strain energy...
   U = 0.5 * np.dot((u.T * K), u)
   
   return dofs, U



def pFEMsol2(p,case):
   h = 0.2
   nodes = np.array([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]])
   elems = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
   bcs = [[0, 4*p + 1], [0.0, 0.0]]
   load = [[0, 4*p + 1], [0, 0]]
   nnodes, dpn = nodes.shape                 # node count and dofs per node
   dofs = dpn * nnodes + 5*(p-1)             # total number of dofs 
   material = tuple([1, 1])
   gauss = grule(p+1)

   K = sp.lil_matrix((dofs, dofs))
   F = np.zeros(dofs)

   #-- Assembling stiffness matrix and force vector...
   for e, conn in enumerate(elems):
      # coordinate array for the element
      X = nodes[conn]
      ldofs = dpn * len(conn) + (p-1)
      k = np.zeros((ldofs, ldofs))
      f = np.zeros(ldofs)

      # element degree of freedom
      if p == 1:
          eft = np.array([dpn * n + i for n in conn for i in range(dpn)])
      if p > 1:
         if e == 0:
            eft = np.array([dpn * n for n in range(ldofs)])
         elif e > 0:
            eft_0 = np.array([1+p*(e-1), 1+p*e])
            eft_1 = np.array([1 + p*e + (n+1) for n in range(p-1)])
            eft = np.append(eft_0, eft_1)

      # derive element k matrix
      for i, xi in enumerate(gauss.xi):
          phi, dphi = shapes(xi, p)
          j = h/2; Jinv = 1/j
          B = np.dot(dphi,Jinv)
          BB = np.kron(B.T,np.identity(dpn))          
          matDT = constitutive(material, dpn)
          k += gauss.wgt[i] * j * np.dot(np.dot(BB.T, matDT), BB)

      # assemble global K matrix
      K[eft[:, np.newaxis], eft] += k

      # derive body force vector f
      bodyf = grule(p+3)
      for i, xi in enumerate(bodyf.xi):
          phi, dphi = np.array(shapes(xi,p))
          Xxi = np.dot(X.T,[phi[0],phi[1]]) 
          j = h/2
          matDT = constitutive(material, dpn)
          f += bodyf.wgt[i] * j * phi * BodyF(matDT, Xxi, 2)

      # assemble global body force vector
      F[eft] += f
      #print('F', F)

   #-- Applying boundary conditions...
   zero = bcs[0]  
   F -= K[:, zero] * bcs[1] 
   K[:, zero] = 0;
   K[zero, :] = 0;  
   K[zero, zero] = 1
   F[zero] = bcs[1]  

   # apply loads
   F[load[0]] += load[1]

   #-- Solving system of equations...
   u = spsolve(K.tocsr(), F)

   #-- Calculating strain energy...
   U = 0.5 * np.dot((u.T * K), u)
   
   return dofs, U

 