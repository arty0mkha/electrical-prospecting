import scipy as sp
import numpy as np

def Ri(m,i,param):
  N = int(len(param)/2+1)
  rho = param[0::2]
  h = param[1::2]
  if(i==N-1):
    return 1.0
  else:
    Rpl1 = Ri(m,i+1,param)
    rr = (rho[i+1]*Rpl1 - rho[i])/(rho[i+1]*Rpl1 + rho[i]) * np.exp(-2*m*h[i])
    return (1+rr)/(1-rr)

def R(m,param):
  return Ri(m,0,param)

def potential_intergrand(m, r, param):
  return (R(m,param) - 1)*sp.special.j0(r*m)

def field_intergrand(m, r, param):
  return m *(R(m, param) - 1)*sp.special.j1(r*m)

def weber_lipchitz(r,z):
  return 1/np.sqrt(np.square(r)+np.square(z))

def weber_lipchitz_derivative(r,z):
  return -r/(np.sqrt(np.square(r)+np.square(z))*(np.square(r)+np.square(z)))

def calculate_apparent_resistance(param, method,r,col=100):
  rho = param[0::2]
  h = param[1::2]

  if method == "U":
    list_bessel0_zeros = np.array([0])
    list_bessel0_zeros = np.append(list_bessel0_zeros, sp.special.jn_zeros(0, col))
    result = 0
    for i in range(col):
      result += r*rho[0]*sp.integrate.quad(potential_intergrand, list_bessel0_zeros[i]/r, list_bessel0_zeros[i+1]/r, args=(r, param))[0]
    result+=r*rho[0]*weber_lipchitz(r,0)
      
  elif method == "E":
    list_bessel1_zeros = np.array([0])
    list_bessel1_zeros = np.append(list_bessel1_zeros, sp.special.jn_zeros(1, col))
    result = 0
    for i in range(col):
      result += r**2*rho[0]*sp.integrate.quad(field_intergrand, list_bessel1_zeros[i]/r, list_bessel1_zeros[i+1]/r, args=(r, param))[0]
    result += -r**2*rho[0]*weber_lipchitz_derivative(r,0)
  return result

if __name__!='__main__':
  print('direct_problem was imported')
