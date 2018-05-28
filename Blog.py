
"""
Blog 23 Apr:

    Question 1.5 unfinihsed
    Test individual P2 okay, haven't added into Problem1_1 yet 
    number of gauss points for body force need to be redifined
    next time: think about analytically how it should be defined
    and put it into the code

Blog 24 Apr:
       
    -------------question to aks during thursday lecture-------------
    1. Question 1.5, for non-smooth problem, shouldn't it be algebraic 
    convergence for both h-FEM and p-FEM?
    
    2. number of gauss point for the body force affects the result 
    a lot, for example using p+3 gauss point, it seems that the strain
    energies are closer to the exact value, but the relation curve of
    error and dofs looks not that nice, however when using p+2 gauss 
    point, although there are larger error, the relation curve between
    error and dofs looks nicer.
    -----------------------------------------------------------------
    
Blog 28 Apr:
    
    Question 1.5 has an issue. For the non-smooth problem, it shouldn't 
    be exponational convergence.
    
    
Blog 3 May: 
    
    p-FEM code checking: 
        - in textpFEM.py, I checked the p-FEM code with p=2,
          by using a exact solution from a quadratic function [u=x(1-x)]. (U = 0.041666667)
          With higher p, the computed results are still equal to 
          the exact as we supposed.
        - Then a cubic solution problem u = -1/6 x^3 + 1/6 x is used U = 0.011111111
          to check p-FEM code with p = 3. 
        - With above check, I supposed my p-FEM code should be correct
    

"""

