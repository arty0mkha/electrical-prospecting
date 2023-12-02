import scipy as sp
import numpy as np

def Ri(m: float,
       i: int,
       param: list
       ) -> float:
  ''' Возвращает значение функции R_i, зависящей от параметров среды, при некотором m 
    
    Parameters
    ----------
    param: numpy.ndarray
      Массив параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    m: float
      Аргумент
    i: int               
      Номер слоя
  '''
  N=int((len(param)+1)/2)
  rho = param[0::2]
  thickness = param[1::2]
  if (i == N-1):
    return 1.0
  else:
    Rpl1 = Ri(m,i+1,param)
    rr = (rho[i+1]*Rpl1 - rho[i])/(rho[i+1]*Rpl1 + rho[i]) * np.exp(-2*m*thickness[i])
    return (1+rr)/(1-rr)

def R(m: float,
      param: np.ndarray
      ) -> float:
  '''
  
  Parameters
  ---------
  m: float
    Аргумент 
  param: List
    Список параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1

  '''
  return Ri(m,0,param)

def potential_intergrand(m: float,
                         r: float,
                         param: np.ndarray
                         ) -> float:
  ''' Возвращает значение подынтегрального выражения, при расёте кажущегося сопротивления через потенциал

    Parameters
    ----------
    param:  list
      Список параметров среды формой (2N-1), N -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    r: float   
      Полуразнос    
    m: float
      Аргумент
   
  '''
  return (R(m,param) - 1)*sp.special.j0(r*m)

def field_intergrand(m: int,r: int,param: list):
  ''' Возвращает значение подынтегрального выражения, при расчёте кажущегося сопротивления через поле

    Parameters
    ----------
    param: numpy.ndarray
        Массив параметров среды формой (2N-1), N -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    m: float
      аргумент
        -
    r: float   
        полуразнос   
  '''
  return m *(R(m, param) - 1)*sp.special.j1(r*m)

def weber_lipchitz(r: float,
                   z: float
                   ) -> float:
  '''
  Возвращает значение интеграла Вебера-Липшица 

  Parameters
  ----------
  r: float
    Полуразнос
  z: float
    Глубина
  '''
  return 1/np.sqrt(np.square(r)+np.square(z))

def weber_lipchitz_derivative(r: float,
                              z: float
                              ) -> float:
  '''
  Возвращает значение производной интеграла Вебера-Липшица 

  Parameters
  ----------
  r: float
    Полуразнос
  z: float
    Глубина
  '''
  return -r/(np.sqrt(np.square(r) + np.square(z))*(np.square(r) + np.square(z)))

def calculate_apparent_resistance(param: list,
                                  method: str,
                                  r: float,
                                  num_of_zeros: int,
                                  ) -> float:
  '''Вычисляет кажущееся сопротивление среды по полю или по потенциалу при фиксированном r

    Parameters
    ----------
    param: numpy.ndarray
      Список параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    method: str
       Переменная отвечающая за выбор по полю ("E") или потенциалу ("U") будет считаться ro кажущееся
    r: float   
      Полуразнос  
    num_of_zeros: int
      Количество нулей бесселя, определяет длину промежутка интегрирования
  '''
  rho = param[0::2]

  if method == "U":
    list_bessel0_zeros = np.array([0])
    list_bessel0_zeros = np.append(list_bessel0_zeros, sp.special.jn_zeros(0, num_of_zeros))
    result = 0
    for i in range(num_of_zeros):
      result += r*rho[0]*sp.integrate.quad(potential_intergrand, list_bessel0_zeros[i]/r, list_bessel0_zeros[i+1]/r, args=(r, param))[0]
    result += r*rho[0]*weber_lipchitz(r,0)
      
  elif method == "E":
    list_bessel1_zeros = np.array([0])
    list_bessel1_zeros = np.append(list_bessel1_zeros, sp.special.jn_zeros(1, num_of_zeros))
    result = 0
    for i in range(num_of_zeros):
      result += r**2*rho[0]*sp.integrate.quad(field_intergrand, list_bessel1_zeros[i]/r, list_bessel1_zeros[i+1]/r, args=(r, param))[0]
    result += -r**2*rho[0]*weber_lipchitz_derivative(r,0)
  return result

if __name__!='__main__':
  print('direct_problem was imported')
