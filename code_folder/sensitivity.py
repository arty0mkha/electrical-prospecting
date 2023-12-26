import numpy as np
import copy
from tqdm.notebook import tqdm
if __name__  == '__main__':
    import direct_problem as direct
else:
    from code_folder import direct_problem as direct

def sense_one_point(r:float, param:list, change_num:int, delta:float, method:str):
    '''Возвращает чувствительность кажущегося сопротивления, посчитанного по method при изменении параметра change_num на delta относительно param
    
    Parameters
    ----------
    method: str
        Способ рачёта кажущеегося сопротивления: 'U' посчитанное через разность потенциалов, 'E' - через электрическое поле   
    '''
    changed_param=copy.copy(param)
    changed_param[change_num]+=delta
    resistance0=direct.calculate_apparent_resistance(param,method,r,10*int(1+r/200))
    resistance1=direct.calculate_apparent_resistance(changed_param,method,r,10*int(1+r/200))
    return param[change_num]/resistance0*(resistance1-resistance0)/(delta)

def sense_line(r_array:np.array, param:list, change_num:int, delta:float, method:str):
    

    n=[]
    for r_i in r_array:
        n.append(sense_one_point(r_i,param,change_num,delta,method))
    return n

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
      drho.append((10)**(i/int(size/2)))
    for i in range(0,int(size/2)+1):
      drho.append((10)**(i/int(size/2)))

  if dh==None:
    dh = []
    for i in range(-int(size/2),0):
      dh.append((10)**(i/int(size/2)))
    for i in range(0,int(size/2)+1):
      dh.append((10)**(i/int(size/2)))

  func_param = []
  for i in range(size):
    g = []
    for j in range(size):
      g.append(parameters[:2*layer_num-2]+[drho[i]*parameters[2*layer_num-2]]+[dh[j]*parameters[2*layer_num-1]]+parameters[2*layer_num:])
    func_param.append(g)
  func_param = np.array(func_param)

  resist = []
  for i in tqdm(range(size)):
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

if __name__  != '__main__':
    print('sensivity was imported')
