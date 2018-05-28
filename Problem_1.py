import numpy as np
import matplotlib.pyplot as plt
from FEMsolver import FEMsol, pFEMsol, pFEMsol2
from writefile import tojson
import math 
from math import log
from scipy.optimize import brentq
#from aposteriori import fn

# initialization
U_l = [];    U_q = [];                     U_q_2 = [];                       U_l_2 = [];
dofs_l = []; dofs_q = [];   dofs_p = [];   dofs_q_2 = [];   dofs_p_2 = [];   dofs_l_2 = [];
er_l = [];   er_q = [];     er_p = [];     er_q_2 = [];     er_p_2 = [];     er_l_2 = []; 
h_l = [];    h_q = [];                     h_q_2 = [];                       h_l_2 = [];

Uexa = 0.0408777548
Uexa2 = 25.138142063

# %% question 1.1:
# using linear lagrange elements
# calculate strain energy & dofs with different mesh size 
print('question 1.1, h-FEM p=1 for smooth problem')
case = 1
for i in range(5):
    nelem = 2**(i+1)
    #print('numer of element',nelem)
    et = 'linear'
    tojson(nelem,et)
    file = "h{}elem-l.json".format(nelem)
    U,dofs,h = FEMsol(file,case)
    er= np.sqrt((Uexa-U)/Uexa)
    U_l.append(U); dofs_l.append(dofs);h_l.append(h)
    er_l.append(er)

C_rate_l = (math.log(er_l[-1])-math.log(er_l[0]))/(math.log(h_l[-1])-math.log(h_l[0]))
C_rate_l = round(C_rate_l,2)

# question 1.1:
# using quadratic lagrange elements
# calculate strain energy & dofs with different mesh size 
print('question 1.1, h-FEM p=2 for smooth problem')
case = 1
for i in range(5):
   nelem = 2**(i+1)
   et = 'quadratic'
   tojson(nelem, et)
   file = "h{}elem-Q.json".format(nelem)
   U,dofs,h= FEMsol(file,case)
   er= np.sqrt((Uexa-U)/Uexa)
   U_q.append(U);dofs_q.append(dofs); h_q.append(h)
   er_q.append(er)

C_rate_q = (math.log(er_q[-1])-math.log(er_q[0]))/(math.log(h_q[-1])-math.log(h_q[0]))
C_rate_q = round(C_rate_q,2)

# plot
plt.figure(1)

plt.loglog(h_l,er_l,'o-',h_q,er_q,'r^-')
plt.xlabel('log(h)')
plt.ylabel('log(e)')
plt.title('Error & Mesh Size (Type 1)')
plt.text(10**(-1),10**(-3), r'rate = {}'.format(C_rate_q))
plt.text(10**(-1),7.5*10**(-2), r'rate = {}'.format(C_rate_l))
plt.legend(('h-FEM p=1', 'h-FEM p=2'),
           shadow=False, loc=(0.72, 0.015))
plt.show()

plt.figure(2)
plt.loglog(dofs_l,er_l,'o-',dofs_q,er_q,'r^-')
plt.legend(('h-FEM p=1', 'h-FEM p=2'),
           shadow=False, loc=(0.008, 0.015))
plt.xlabel('log(N)')
plt.ylabel('log(e)')
plt.title('Error & DOFs (Type 1)')
plt.text(3,2.2*(10)**(-1),3)
plt.text(5,1*(10)**(-1),5)
plt.text(9,6*(10)**(-2),9)
plt.text(17,2.5*(10)**(-2),17)
plt.text(33,1.3*(10)**(-2),'DOFs = 33')

plt.text(5,4*(10)**(-3),5)
plt.text(9,1*(10)**(-3),9)
plt.text(17,2*(10)**(-4),17)
plt.text(33,2*(10)**(-4),33)
plt.text(40,5*(10)**(-5),'DOFs = 65')
plt.show()

#%% question 1.2: using p-FEM, 2elements, from p=1 to 5
print('question 1.2, p-FEM')
case = 1
for i in range(5):
    p = i + 1
    p_dofs, p_U = pFEMsol(p,case)
    p_er= np.sqrt(abs(Uexa-p_U)/U)
    dofs_p.append(p_dofs)
    er_p.append(p_er)

# plot
plt.figure(3)
plt.loglog(dofs_l,er_l,'o-',dofs_q,er_q,'r^-',dofs_p,er_p,'g-.*')
plt.legend(('h-FEM p=1', 'h-FEM p=2','p-FEM h=0.5'),
           shadow=False, loc=(0.008, 0.015))
plt.text(2.8,2.5*(10)**(-1),3)
plt.text(4.8,3*(10)**(-3),5)
plt.text(6.5,4*(10)**(-4),7)
plt.text(9.5,4*(10)**(-5),9)
plt.text(11.5,0.15*(10)**(-4),'DOFs = 11')
plt.xlabel('log(N)')
plt.ylabel('log(e)')
plt.title('Error & DOFs (Type 1)')
plt.show()

#%% question 1.3: a posteriori error estimation 
# using data points (er, N) from h-FEM

print('question 1.3, posteriori error estimation ')

Q = np.log(dofs_l[1]/dofs_l[0])/np.log(dofs_l[2]/dofs_l[1])

def fn(UX):
   return (UX - U_l[0])/(UX - U_l[1]) - ((UX - U_l[1])/(UX - U_l[2]))**Q

root = brentq(fn,0.0404,0.0410,full_output=True)
print('Approximated value of energy:',root[0])

#%% question 1.4:
print('question 1.4, h-FEM for nearly non-smooth problem')
case = 2
for i in range(4): 
    nelem =  5*2**i
    #print('numer of element',nelem)
    et = 'linear'
    tojson(nelem,et)
    file = "h{}elem-l.json".format(nelem)
    U,dofs,h = FEMsol(file,case)
    er= np.sqrt(abs(Uexa2-U)/Uexa2)
    U_l_2.append(U); dofs_l_2.append(dofs);h_l_2.append(h)
    er_l_2.append(er)

C_rate_l_2 = (math.log(er_l_2[-1])-math.log(er_l_2[-2]))/(math.log(h_l_2[-1])-math.log(h_l_2[-2]))
C_rate_l_2 = round(C_rate_l_2,2)

case = 2
for i in range(4):
   nelem = 5*2**i
   et = 'quadratic'
   tojson(nelem, et)
   file = "h{}elem-Q.json".format(nelem)
   U_2,dofs_2,h_2 = FEMsol(file,case)
   er_2= np.sqrt(abs(Uexa2-U_2)/Uexa2)
   U_q_2.append(U_2);dofs_q_2.append(dofs_2); h_q_2.append(h_2)
   er_q_2.append(er_2)

C_rate_q_2 = (math.log(er_q_2[-1])-math.log(er_q_2[-2]))/(math.log(h_q_2[-1])-math.log(h_q_2[-2]))
C_rate_q_2 = round(C_rate_q_2,2)

# plot
plt.figure(4)

plt.subplot(121)
plt.loglog(h_l_2, er_l_2,'o-',h_q_2,er_q_2,'r^-')
plt.xlabel('log(h)')
plt.ylabel('log(e)')
plt.title('Error & Mesh Size (Type 2)')
plt.text(10**(-2),10**(-1), r'rate = {}'.format(C_rate_l_2))
plt.text(10**(-2),10**(-2), r'rate = {}'.format(C_rate_q_2))
#plt.legend(('h-FEM p=1','h-FEM p=2',),
#           shadow=False, loc=(0.72, 0.015))
#plt.show()

plt.subplot(122)
plt.loglog(dofs_l_2,er_l_2,'o-',dofs_q_2,er_q_2,'r^-')
plt.legend(('h-FEM p=1','h-FEM p=2',),
           shadow=False, loc=(0.008, 0.015))
plt.xlabel('log(N)')
plt.ylabel('log(e)')
plt.title('Error & DOFs (Type 2)')
#plt.text(6,7*(10)**(-1),6)
#plt.text(11,8.2*(10)**(-1),11)
#plt.text(21,5.8*(10)**(-1),21)
#plt.text(41,3.3*(10)**(-1),'DOFs = 41')

#plt.text(11,4.5*(10)**(-1),11)
#plt.text(21,2.5*(10)**(-1),21)
#plt.text(38,8*(10)**(-2),41)
#plt.text(55,3*(10)**(-2),'DOFs = 81')
plt.show()

#%% question 1.5:
print('question 1.5, p-FEM for nearly non-smooth problem')
case = 2
for i in range(5):
    p = i + 1
    p_dofs, p_U = pFEMsol2(p,case)
    p_er= np.sqrt(abs(Uexa2-p_U)/Uexa2)
    dofs_p_2.append(p_dofs)
    er_p_2.append(p_er)
    
# plot
plt.figure(6)
plt.loglog(dofs_l_2,er_l_2,'o-',dofs_q_2,er_q_2,'r^-',dofs_p_2,er_p_2,'g-.*')
plt.legend(('h-FEM p=1','h-FEM p=2','p-FEM h=0.2'),
           shadow=False, loc=(0.008, 0.015))
plt.text(5.5,6.1*(10)**(-1),6)
plt.text(10.1,4.5*(10)**(-1),11)
plt.text(15,3.3*(10)**(-1),16)
plt.text(19.2,2*(10)**(-1),21)
plt.text(26,6*(10)**(-2),'DOFs = 26')
plt.xlabel('log(N)')
plt.ylabel('log(e)')
plt.title('Error & DOFs (Type 2)')
plt.show()   