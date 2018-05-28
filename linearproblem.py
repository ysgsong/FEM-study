from gauss import quadrature
import sys, os, json, inspect
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from shape_fns import fns_l2, fns_l3, jacobian
from bodyforce import BodyF
from writefile import tojson
import matplotlib.pyplot as plt
from elastic import constitutive

Uexa = 0.03125
error = []
meshsi = []
et = 'linear'

for a in range(5):

  tojson(a+1,et)
  file = "{}elem.json".format(a+1)

  with open(file) as f:
      print("-- Reading file '{}'".format(file))
      data = json.load(f)                       # load
      nodes = np.array(data['nodes'])           # nodes array to numpy array
      elems = np.array(data['elements'])        # elements array to numpy array
      nnodes, dpn = nodes.shape                 # node count and dofs per node
      etype = data['etype']
      bcs = data['boundary']
      h = data['meshsize']
      meshsi.append(h)
      load = data['load']
      dofs = dpn * nnodes                       # total number of dofs
      material = tuple(data['material'])
      gauss = quadrature(etype, data['gauss'])  # quadrature data structure

  K = sp.lil_matrix((dofs, dofs))
  print(K.toarray())
  F = np.zeros(dofs)
  print(F)

  print("-- Assembling stiffness matrix and force vector...")
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
      print('k',k)

      # assemble global K matrix
      K[eft[:, np.newaxis], eft] += k

      bodyforce = quadrature(etype, 4)
      for i, xi in enumerate(bodyforce.xi):
          N, dN = eval('fns_{}'.format(etype))(xi, X)
          Jinv, j = jacobian(X, dN)
          matDT = constitutive(material, dpn)
          f += bodyforce.wgt[i] * j * N * BodyF(matDT,xi)

      # assemble global body force vector
      F[eft] += f
      print('F', F)

  print('K', K.toarray())

  print("-- Applying boundary conditions...")
  zero = bcs[0]               # array of rows/columns which are to be zeroed out
  F -= K[:, zero] * bcs[1]    # modify right hand side with prescribed values
  K[:, zero] = 0;
  K[zero, :] = 0;             # zero-out rows/columns
  K[zero, zero] = 1           # add 1 in the diagonal
  F[zero] = bcs[1]            # prescribed values

  # apply loads
  F[load[0]] += load[1]
  print('F finally', F)
  print('K finally\n', K.toarray())

  print("-- Solving system of equations...")
  u = spsolve(K.tocsr(), F)
  print('u is', u)

  print("-- Calculating strain energy...")
  U = 0.5 * np.dot((u.T*K), u)
  print(U)

  #er = np.sqrt(abs(Uexa - U) / Uexa)
  #error.append(er)

#print(error)


#plt.loglog(meshsi,error,'b--')



