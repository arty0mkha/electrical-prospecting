import scipy as sp
import numpy as np
if __name__  == '__main__':
    import direct_problem as direct
else:
    from code import direct_problem as direct

# Целевые функции

def RMSE(calculated_data: np.ndarray,
         refernce_data: np.ndarray
         ) -> float:
    ''' Возвращает RMSE между data и reference_data
    
    Parameters
    ----------
    calculated_data: numpy.ndarray
        Массив данных длиной K
    reference_data: numpy.ndarray
        "Эталонный" массив данных длиной K
    '''
    K = calculated_data.shape[0]
    # считаем сумму квадратов разности значений 
    s = sum(np.square(calculated_data-refernce_data))
    # возвращаем RMSE
    return np.sqrt(s/K)

#Обратная задача с использованием решения прямой задачи

def Loss_direct(param: list,
         loss_type : str,
         function_type: str,
         data: np.ndarray
         ) -> float:
    ''' Возвращает значение ошибки loss_type для среды с параметрами param и данных data, полученных для function_type
    
    Parameters
    ----------
    param: list
        Массив параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    loss_type: str
        Тип целевой функции
    function_type: str
        Тип минимизируемой функции: 'rhoa' - кажущееся сопротивление, 'u' - разность потенциалов, 'E' - электрическое поле
    data: numpy.ndarray
        Массив формы (K,2), K = количество точек, data[i]=[r_i, f_i], r_i - полуразнос, f_i -измеренное значение    
    '''
    direct_data = []
    for r_i in data[:, 0]:
        direct_data.append(direct.calculate_apparent_resistance(param, function_type, r_i,num_of_zeros=10*int(1+r_i/200)))

    direct_data = np.array(direct_data)
    
    if loss_type == 'RSME':
            return RMSE(direct_data, data[:,1])

def inverse_problem_solver(N_list : list,
                    function_type : str,
                    data : np.ndarray,
                    minimization_method : str = 'COBYLA',
                    loss_type : str = 'RSME',
                    thickness_max : float =5*10**2,
                    tolerance : float = 10**(-5),
                    start: list =[],
                    boundaries: list =[]
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
    start: list
        список из моделей среды для каждой из n_list 
    boundaries: list
        Список из кортежей определяющий границы значений параметров среды
    '''
    # создание списков подобранных моделей и ошибокs
    results_list = []
    results_losses = []

    # ограничение на сопротивление слоёв
    rhoa_max = max(data[:][1])
    for i in range(len(N_list)):
        # Создание ограничений на rhoa, thickness для каждого слоя в scipy.optimize.minimize
        if boundaries == []:
            for j in range(N_list[i]):
                boundaries.append((0,2*rhoa_max))
                boundaries.append((0,thickness_max))
            boundaries = tuple(boundaries[:-1])
            start_param=start[i]
            # минимизация
            result = sp.optimize.minimize(fun = Loss_direct,
                                        x0 = start_param,
                                        args = (loss_type, function_type, data),
                                        method = minimization_method,
                                        bounds = boundaries, 
                                        tol = tolerance
                                      )
        else:
            start_param=start[i]
            # минимизация
            result = sp.optimize.minimize(fun = Loss_direct,
                                        x0 = start_param,
                                        args = (loss_type, function_type, data),
                                        method = minimization_method,
                                        bounds = boundaries[i], 
                                        tol = tolerance
                                      )
        
        # подобранные параметры записываются в список
        results_list.append(result)
        
        # ошибка записывается в список
        results_losses.append(result.fun)

    # возвращается модели и номер с минимальным значением ошибки loss_type
    return results_list, np.where(results_losses == np.min(results_losses))[0][0]       

if __name__  != '__main__':
    print('inverse_problem was imported')