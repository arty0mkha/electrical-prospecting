import numpy as np
from code import direct_problem as direct

def logderivative(func:np.array, var:np.array, length:int) -> np.ndarray:
  """
  Возвращает логарифмическую производную

  Параметры:
  ----------
  func:np.ndarray
  Массив значений функции
  var:np.ndarray
  Массив значений переменной
  length:int
  Длина массива производных, не больше минимальной длины из принимаемых массивов
  """
  logderiv = []
  for i in range(length-1):
    logderiv.append(var[i]/func[i]*(func[i+1]-func[i])/(var[i+1]-var[i]))
  return (np.array(logderiv))

def matrix_models(parameters:list, layer_num:int, size:int, method:str, dh:list=None, drho:list=None) -> np.ndarray:
  """
  Возвращает матрицу ранга size, содержающие высчитанные в прямой задаче модели
  параметры, изменяемые с шагом dh и drho

  Параметры:
  ----------
  parameters:list
  Список входных параметров
  layer_num:int
  Номер слоя, который подвергается (XD подвергается :D) изменению
  size:int
  Размер получаемой матрицы
  method:str
  Метод вычисления: по потенциалу(U) или по полю(E)
  dh:list
  Список высчитанных шагов по оси Ох (необязателен)
  drho:list
  Список высчитанных шагов по оси Оy (необязателен)
  """

  r=np.logspace(-1,2,100)

  if drho==None:
    drho = []
    for i in range(-int(size/2),0):
      drho.append((3/2)**(np.abs(i)/int(size/2)))
    for i in range(0,int(size/2)+1):
      drho.append((3/2)**(i/int(size/2)))

  if dh==None:
    dh = []
    for i in range(-int(size/2),0):
      dh.append((3/2)**(np.abs(i)/int(size/2)))
    for i in range(0,int(size/2)+1):
      dh.append((3/2)**(i/int(size/2)))

  func_param = []
  for i in range(size):
    g = []
    for j in range(size):
      g.append(parameters[:2*layer_num-2]+[drho[i]*parameters[2*layer_num-2]]+[dh[j]*parameters[2*layer_num-1]]+parameters[2*layer_num:])
    func_param.append(g)
  func_param = np.array(func_param)

  resist = []
  for i in range(size):
    resistance=[]
    for j in range(size):
      for r_i in r:
        resistance.append(direct.calculate_apparent_resistance(func_param[i][j],method,r_i,10*int(1+r_i/200)))
    resist.append(np.array(resistance))
  res = []
  for i in range(size):
    a = []
    for j in range(size):
      a.append(resist[i][j*100:100*(j+1)])
    res.append(np.array(a))
  return np.array(res)
