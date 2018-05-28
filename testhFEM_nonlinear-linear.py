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
import math


Uexa = 0.0408777548
error = []
meshsi = []
Uall = []

for a in range(1):

  nelem = 2**(a+1)

  et = 'linear'
  tojson(nelem,et)
  file = "h{}elem-l.json".format(nelem)

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
  F = np.zeros(dofs)

  print("-- Assembling stiffness matrix and force vector...")
  for e, conn in enumerate(elems):
      # coordinate array for the element
      X = nodes[conn]
      ldofs = dpn * len(conn)
      k = np.zeros((ldofs, ldofs))
      f = np.zeros(ldofs)
      eft = np.array([dpn * n + i for n in conn for i in range(dpn)])
      print(eft)
      print(X)
      # derive element k matrix
      for i, xi in enumerate(gauss.xi):
          print(xi)
          N, dN = eval('fns_{}'.format(etype))(xi, X)
          print('N:',N)
          Jinv, j = jacobian(X, dN)
          B = np.dot(dN, Jinv)
          BB = np.kron(B.T, np.identity(dpn))
          matDT = constitutive(material, dpn)
          k += gauss.wgt[i] * j * np.dot(np.dot(BB.T, matDT), BB)

      # assemble global K matrix
      K[eft[:, np.newaxis], eft] += k

      bodyf = quadrature(etype, 3)
      for i, xi in enumerate(bodyf.xi):
          N, dN = eval('fns_{}'.format(etype))(xi, X)
          Xxi = np.dot(X.T,N)  #map xi back to X and then sustitude into the bodyforce!!!!
          Jinv, j = jacobian(X, dN)
          matDT = constitutive(material, dpn)
          f += bodyf.wgt[i] * j * N * BodyF(matDT,Xxi,1)

      # assemble global body force vector
      F[eft] += f
      #print('F', F)

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

  print("-- Calculating strain energy...")
  U = 0.5 * np.dot((u.T*K), u)
  Uall.append(U)
#  er = np.sqrt((Uexa - U) / Uexa)
#  error.append(er)

print(Uall)
#print(error)
#print(meshsi)

#C_rate = (math.log(error[-1])-math.log(error[0]))/(math.log(meshsi[-1])-math.log(meshsi[0]))
#print('Congervence rate = ',C_rate)

#plt.loglog(meshsi, error, 'b--')
#plt.xlabel('log(h)')
#plt.ylabel('log(e)')
#plt.title('Log-Log plot of Error & Mesh Size relation')
#plt.show()


# 改动：
# 1. EA: 从"writefile.py"里面改，data["material"]= [1.0, 2.0] 后面两个值分别是E和A
# 2. L: 从"writefile.py"里面改2各地方，一个是nodal value, 一个是meshsize
# 3. Bodyforce: 从"bodyforce.py"里面改 du^2/d^2x
# 4. 如果增加了外界load，千万记得到"writefile.py"中改load
#

