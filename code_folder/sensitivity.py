import numpy as np
import copy
from tqdm.notebook import tqdm
if __name__  == '__main__':
    import direct_problem as direct
    import inverse_problem as inv
else:
    from code_folder import direct_problem as direct
    from code_folder import inverse_problem as inv


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


def matrix_models(r:np.ndarray, parameters:list, layers_change:tuple, size:int, method:str, dh:list=None, drho:list=None) -> np.ndarray:
  """
  Возвращает матрицу ранга size, содержающие высчитанные в прямой задаче модели
  параметры, изменяемые с шагом dh и drho

  Параметры:
  ----------
  r: np.array
    Полуразнос
  parameters:list
    Список входных параметров
  layers_change:tuple
    Кортеж номеров параметров, которые подвергаются (XD подвергаются :D) изменению на dh или drho
  size:int
    Размер получаемой матрицы
  method:str
    Метод вычисления: по потенциалу(U) или по полю(E)
  dh:list
    Список высчитанных шагов по оси Ох (необязателен)
  drho:list
    Список высчитанных шагов по оси Оy (необязателен)
  """

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
      parameters_g = []
      for k in range(len(parameters)):
        if k==layers_change[1] and k%2!=0:
          parameters_g.append(parameters[k]+dh[j])
        if k==layers_change[0] and k%2==0:
          parameters_g.append(parameters[k]+drho[i])
        else:
          parameters_g.append(parameters[k])
      g.append(parameters_g)
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
      a.append(resist[i][j*r.shape[0]:r.shape[0]*(j+1)])
    res.append(np.array(a))
  return np.array(res)


def error_map_by_Artyom(r:np.ndarray,
                        param_:list, 
                        change_indexes_:list, 
                        size:int, 
                        method:str, 
                        change: float=10):
    '''  Строит карту ошибок относительно выбранной модели среды при изменении двух выбранных параметров  
    
    Parameters
    ----------
    r: np.array
        Полуразнос
    param_: list
        Список параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    change_indexes_: int
        Отвечает за выбор изменяемых параметров
    size: int
        Определяет длину стороны картинки в пикселях
    method: str
        Переменная отвечающая за выбор по полю ("E") или потенциалу ("U") будет считаться кажущееся сопротивление
    change: float
        Определяет изменение параметро - от /change до *change
    '''

    etalon_resistance = np.zeros(shape=r.shape[0])
    for k in range(r.shape[0]):
        etalon_resistance[k] = direct.calculate_apparent_resistance(param_, method, r[k], 10+int(r[k]/200))

    local_param = copy.copy(param_)
    new_first_param = np.logspace(np.log10(param_[change_indexes_[0]]/change), np.log10(param_[change_indexes_[0]]*change), size)
    new_second_param = np.logspace(np.log10(param_[change_indexes_[1]]/change), np.log10(param_[change_indexes_[1]]*change), size)
    
    loss_map = np.zeros(shape=(size, size))

    for i in range(size):
        for j in range(size):
            local_param[change_indexes_[0]] = new_first_param[i] 
            local_param[change_indexes_[1]] = new_second_param[j] 
            
            current_resistance = np.zeros(shape=r.shape[0])
            for k in range(r.shape[0]):
                current_resistance[k]=direct.calculate_apparent_resistance(local_param, method, r[k], 10+int(r[k]/200))

            loss_map[i][j]=inv.RMSPE(current_resistance, etalon_resistance)

    return loss_map


if __name__  != '__main__':
    print('sensivity was imported')
