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
  return R(m,param) * sp.special.j0(r * m)

def field_intergrand(m, r, param):
  return m * R(m, param) * sp.special.j1(r * m)

def calculate_apparent_resistance(param, method,r):
  rho = param[0::2]
  h = param[1::2]
  if method == "U":
    result=[]
    for r_i in r:
      result.append( r_i*rho[0]*sp.integrate.quad(potential_intergrand, 0, np.inf, args=(r_i, param))[0])
  elif method == "E":
    result=[]
    for r_i in r:
      result.append( r_i**2*rho[0]*sp.integrate.quad(field_intergrand, 0, np.inf, args=(r_i, param))[0])
  return result
if __name__!='__main__':
  print('direct_problem was imported')
