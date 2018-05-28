from elastic import constitutive


def BodyF(matDT,x,case):
    if case == 1:
      d2u_dx2 = -(2*(0.5+0.5**3*0.8*(x-0.2)))/((0.25*(x-0.2)**2+1)**2)
      #d2u_dx2 = -1
    elif case == 2:
       d2u_dx2 = -(2*(50+50**3*(x-0.2)*0.8)/(50**2*(x-0.2)**2+1)**2)
    #d2u_dx2 = -x
    
    f = - matDT*d2u_dx2

    return f
