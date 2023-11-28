import scipy as sp
import numpy as np
if __name__  == '__main__':
    from inverse_problem import RMSE
else:
    from code.inverse_problem import RMSE
#Обратная задача с использованием обобщения двухслойной модели

def rhoa (r : np.ndarray,
          rho1 : float,
          rho2,
          h :float
          ) -> np.ndarray :
    ''' Возвращает значение кажущегося сопротивления на поверхности для двухслойной модели с сопротивлениями rho1, rho2 и мощностью первого слоя h при полуразносе r

    Parameters
    ----------
    r: numpy.ndarray
        Полуразнос установки
    rho1: float
        Сопротивление верхнего слоя
    rho2: float или numpy.ndarray
        Сопротивление нижнего слоя. \n Если массив, то должен быть той же длины, что и r
    h: float
        мощность верхнего слоя   
    '''
    s = 0
    maxJ = 100
    for j in range (1, maxJ) :
        k12 = (rho2-rho1)/(rho2+rho1)
        s +=  r**3*k12**j /(r**2 + (2*j*h)**2)**(3/2)
    return rho1*(1 +2*s)

def aprox_rhoa(r : np.ndarray,
                        param : np.ndarray
                        ) -> np.ndarray:
    ''' Возвращает значение кажущегося сопротивления на поверхности для N-слойной модели param при полуразносе r используя двухслойную модель

    parameters
    ----------
    r: numpy.ndarray
        Полуразнос установки
    param: numpy.ndarray
        Массив параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    '''
    N=int((param.shape[0]+1)/2) 
    if N != 1:
        # Сопротивление слоёв пересчитываются снизу вверх как кажущиеся сопротивления в двух-слойной модели 
        rk=rhoa(r,param[2*((N-2))], param[2*(N-1)], param[2*((N-1))-1])
        for i in range(1,N-1):
            rk=rhoa(r,param[2*((N-2-i))], rk, param[2*(N-2-i)+1])
        # Возвращаем кажущееся сопротивдение на поверхности
        return rk
    else:
        return param[0]

def loss_N_layers(param : np.ndarray,
                  loss_type: str,
                  data : np.ndarray
                  ) -> float:
    ''' Возвращает значение ошибки loss_type для N-слойной модели param и данных data, полученных для кажущегося сопротивления
    
    Parameters
    ----------
    param: numpy.ndarray
        Массив параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1  
    loss_type: str
        Тип целевой функции       
    data: numpy.ndarray
        Массив формы (K,2), K = количество точек, data[i]=[r_i, f_i], r_i - полуразнос, f_i -измеренное значение    
    '''
    if loss_type == 'RSME':
        aprox_data = aprox_rhoa(data[:,0], param)
        # возвращаем RMSE
        return RMSE(data[:,1],aprox_data)

def aprox_inverse_problem_solver(N_list : list,
                    data : np.ndarray,
                    minimization_method : str = 'COBYLA',
                    loss_type : str = 'RSME',
                    thickness_max : float =10**2,
                    tolerance : float = 10**(-5),
                    auto: bool = True,
                    start: list =[]
                    ):
    '''Возвращает list из N_list[i]-слойных моделей в виде объекта класса scipy.optimize.OptimizeResult и индекс модели с минимальной ошибкой
    
    Parameters
    ----------
    N_list: list
        Список из числа слоёв в моделях, среди которых будет происходить подбор наиболее подходящей  
    data: numpy.ndarray
        Массив формы (K,2), K = количество измерений, data[i]=[r_i,f_i], r_i - полуразнос, f_i -измеренное значение
    minimization_method: str, optional
        Метод оптимизации для scipy.optimize.minimize. \n
        Доступные варианты: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr'
    loss_type: str, optional
        Тип целевой функции
    thickness_max: float, optional
        Максимальная мощность слоёв в модели
    tolerance: float, optional
        tolerance для scipy.opnimize.minimize
    auto: bool
        Отвечает за выбор между автоматической генерацией стартовых параметров модели и вводом
    start: list
        список из моделей среды для каждой из n_list 
    '''
    # создание списков подобранных моделей и их ошибок
    results_list = []
    results_losses = []


    # ограничение на сопротивление слоёв
    rhoa_max = max(data[:][1])
    rhoa_min=min(data[:][1])

    for i in range(len(N_list)):
        # Создание ограничений на rhoa, thickness для каждого слоя в scipy.optimize.minimize
        boundaries = []
        for j in range(N_list[i]):
            boundaries.append((0,2*rhoa_max))
            boundaries.append((0,thickness_max))
        boundaries = tuple(boundaries[:-1])
        if not auto:
            start_param=start[i]
        else:
            # Создание начальных значений rhoa, thickness для минимизации
            start_param=np.matmul(np.ones((N_list[i],1)),np.array([[(rhoa_max+rhoa_min)/2, thickness_max/5]])).reshape(-1)[:-1]
        
        # минимизация
        result = sp.optimize.minimize(fun = loss_N_layers,
                                      x0 = start_param,
                                      args = (loss_type, data),
                                      method = minimization_method,
                                      bounds = boundaries, 
                                      tol = tolerance
                                      )
        
        # подобранные параметры записываются в список
        results_list.append(result)
        
        # ошибка записывается в список
        results_losses.append(result.fun)

    # возвращается модели и номер с минимальным значением ошибки loss_type
    return results_list, np.where(results_losses == np.min(results_losses))[0][0]


if __name__  != '__main__':
    print('aprox_inverse_problem was imported')