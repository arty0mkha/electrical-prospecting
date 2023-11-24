import scipy as sp
import numpy as np
import direct_problem as dir

# Целевые функции

def RMSE(calclated_data: np.ndarray,
         refernce_data: np.ndarray
         ) -> float:
    ''' Возвращает RMSE между data и reference_data
    
    Parameters
    ----------
    calculated_data: numpy.ndarray
        Массив данных 
    reference_data: numpy.ndarray
    "Эталонный" массив данных'''
    s = 0
    K = calclated_data.shape[0]
    # считаем сумму квадратов разности значений
    for i in range(K):
        s += np.square(calclated_data[i] - refernce_data[i])
    # возвращаем RMSE
    return np.sqrt(s/K)

#Обратная задача с использованием решения прямой задачи

def Loss_direct(param: np.ndarray,
         loss_type : str,
         function_type: str,
         data: np.ndarray
         ) -> float:
    ''' Возвращает значение ошибки loss_type для среды с параметрами param и данных data, полученных для function_type
    
    Parameters
    ----------
    param: numpy.ndarray
        Массив параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    loss_type: str
        Тип целевой функции
    function_type: str
        Тип минимизируемой функции: 'rhoa' - кажущееся сопротивление, 'u' - разность потенциалов, 'E' - электрическое поле    
    data: numpy.ndarray
        Массив формы (K,2), K = количество точек, data[i]=[r_i, f_i], r_i - полуразнос, f_i -измеренное значение    
    '''
    if loss_type == 'RSME':
        direct_data = dir.direct_problem(function_type, param, data[:, 0]) # direct_problem - функция решающая прямую задачу для function_type в среде param и возвращающая значение function_type в точке r_i
        return RMSE(direct_data, data[:,1])

def inverse_problem_solver(N_list : list,
                    function_type : str,
                    data : np.ndarray,
                    minimization_method : str = 'COBYLA',
                    loss_type : str = 'RSME',
                    thickness_max : float =10**3,
                    tolerance : float = 10**(-5)
                    ):
    '''Возвращает list из N_list[i]-слойных моделей в виде объекта класса scipy.optimize.OptimizeResult и индекс модели с минимальной ошибкой
    
    Parameters
    ----------
    N_list: list
        Список из числа слоёв в моделях, среди которых будет происходить подбор наиболее подходящей  
    function_type: str
        Тип минимизируемой функции: 'rhoa' - кажущееся сопротивление, 'u' - разность потенциалов, 'E' - электрическое поле    
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
        tolerance для scipy.optimize.minimize
    '''
    # создание списков подобранных моделей и ошибокs
    results_list = []
    results_losses = []

    # ограничение на максимальное сопротивление слоёв
    rhoa_max = 2*max(data[:][1])
    
    for N in N_list:

        # Создание ограничений на rhoa, thickness для каждого слоя в scipy.optimize.minimize
        boundaries = []
        for i in range(N):
            boundaries.append((0,rhoa_max))
            boundaries.append((0,thickness_max))
        boundaries = tuple(boundaries[:-1])
        
        # Создание начальных значений rhoa, thickness для минимизации
        start_param=np.matmul(np.ones(shape=(N,1)).T*np.array([[rhoa_max/4, thickness_max/2]])).reshape(-1)
        
        # минимизация
        result = sp.optimize.minimize(fun = Loss_direct,
                                      x0 = start_param,
                                      args = (loss_type, function_type, data),
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
                    tolerance : float = 10**(-5)
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
    '''
    # создание списков подобранных моделей и их ошибок
    results_list = []
    results_losses = []

    # ограничение на максимальное сопротивление слоёв
    rhoa_max = 2*max(data[:][1])
    
    for N in N_list:

        # Создание ограничений на rhoa, thickness для каждого слоя в scipy.optimize.minimize
        boundaries = []
        for i in range(N):
            boundaries.append((0,rhoa_max))
            boundaries.append((0,thickness_max))
        boundaries = tuple(boundaries[:-1])
        # Создание начальных значений rhoa, thickness для минимизации
        start_param=np.matmul(np.ones((N,1)),np.array([[rhoa_max/2, thickness_max/2]])).reshape(-1)[:-1]
        
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

print('inverse_problem was imported')