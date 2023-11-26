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
return R(m,param) * special.j0(r * m)

def field_intergrand(m, r, param):
return m * R(m, param) * special.j1(r * m)

def calculate_resistance(param, method,r):
N = int(len(param) / 2 + 1)
rho = param[0::2]
h = param[1::2]

if method == "U":
result = r*rho[0]*integrate.quad(potential_intergrand, 0, np.inf, args=(r, param))[0]
elif method == "E":
result = r**2*rho[0]*integrate.quad(field_intergrand, 0, np.inf, args=(r, param))[0] 

print('direct_problem was imported')
