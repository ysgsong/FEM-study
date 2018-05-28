# -*- coding: utf-8 -*-
"""
A simple shape function 1-d, gauss-lobatto-polynomials

@author: Yi Zhang. Created on Sat May 12 19:03:06 2018
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft
         Delft, Netherlands
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
# %% Here we define a new calss called Polynomials as our shape functions
class Polynomials(object):
    def __init__(self, nodes):
        """ initialize, when a new instance is defined, this function will be called
            automatically.
        """
        self.nodes = nodes # nodes, the lagrange points which define the polynomials
        self.p = np.size(self.nodes) - 1 # we say the order of the polynomials is the number of nodes minus 1
    
    # %% polynomials
    def evaluate_basis(self, x=None):
        """ here, we evaluate the shape functions determined by self.nodes at 
            points x. Notice that when x is not given, we set x=self.nodes, then
            this method should return a 2-d array identity array.
        """
        if x is None: 
            x = self.nodes
        basis = np.ones((self.p+1, np.size(x)))
        # lagrange basis functions
        for i in range(self.p+1):
            for j in range(self.p+1):
                if i != j:
                    basis[i, :] *= (x - self.nodes[j]) / (self.nodes[i] - self.nodes[j])
        return basis
    
    # %% plt
    def plot_shape_functions(self, plot_density=300, 
                             ylim_ratio=0.15, title="lagrange_basis"):
        """ This is a method to plot the shape_functions.
        """
        x = np.linspace(-1, 1, plot_density)
        basis = self.evaluate_basis(x=x)
        bmx = basis.max(); bmi = basis.min()
        interval = bmx - bmi
        ylim = [bmi - interval*ylim_ratio, bmx + interval*ylim_ratio]
        plt.figure()
        for basis_i in basis:
            plt.plot(x, basis_i, linewidth=1)
        for i in self.nodes:
            plt.plot([i, i], ylim, '--', color=(0.2,0.2,0.2,0.2), linewidth=0.9)
        plt.plot([-1,1], [1,1], '--', color=(0.2,0.2,0.2,0.2), linewidth=0.9)
        plt.plot([-1,1], [0,0], '--', color=(0.2,0.2,0.2,0.2), linewidth=0.9)
        plt.title(title)
        plt.ylim(ylim)
        plt.xlim([-1, 1])
        plt.show()
        
# %% MAIN
if __name__ == "__main__":
    nodes = np.array([-1, -0.5, 0.5, 1])
    my_basis = Polynomials(nodes) # define a new instance of Polynomials, the __init__() is called automatically
    print(my_basis.evaluate_basis())
    my_basis.plot_shape_functions()
    
    
    