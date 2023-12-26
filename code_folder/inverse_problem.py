import scipy as sp
import numpy as np
if __name__  == '__main__':
    import direct_problem as direct
else:
    from code_folder import direct_problem as direct

# Целевые функции

def RMSE(calculated_data:np.ndarray,
         reference_data:np.ndarray
        ) -> np.float64:
  """
  Возвращает целевую функцию от измеренных и теоретических значений

  Параметры:
  ----------
  calculated_data: numpy.ndarray
  Массив данных длиной K
  reference_data: numpy.ndarray
  "Эталонный" массив данных длиной K
  """
  return np.sqrt(np.mean((calculated_data - reference_data)**2))

def RMSPE(calculated_data: np.ndarray,
         refernce_data: np.ndarray
         ) -> float:
    ''' Возвращает RMSEP % между data и reference_data
    
    Parameters
    ----------
    calculated_data: numpy.ndarray
        Массив данных длиной K
    reference_data: numpy.ndarray
        "Эталонный" массив данных длиной K
    '''
    K = calculated_data.shape[0]
    # считаем сумму квадратов разности значений 
    s = sum(np.square((calculated_data-refernce_data)/refernce_data))
    # возвращаем RMSPE %
    return np.sqrt(s/K)*100

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
    
    if loss_type == 'RMSE':
            return RMSE(direct_data, data[:,1])
    if loss_type == 'RMSPE':
        return RMSPE(direct_data, data[:,1])
    

def inverse_problem_solver(function_type: str,
                    data: np.ndarray,
                    start: list,
                    boundaries: list,
                    minimization_method: str ='COBYLA',
                    loss_type: str ='RMSE',
                    tolerance: float =1e-5,
                    ):
    '''Возвращает слоистую модель в виде объекта класса scipy.optimize.OptimizeResult
    
    Parameters
    ----------
    function_type: str
        Тип минимизируемой функции: Кажущееся сопротивление: 'U' посчитанное через разность потенциалов, 'E' - через электрическое поле    
    data: numpy.ndarray
        Массив формы =[r_i,f_i], r_i - полуразнос, f_i -измеренное значение
    start: list
        Список стартовых значений для минимизации loss
    boundaries: list
        Список из кортежей границ значений параметров среды
    minimization_method: str, optional
        Метод оптимизации для scipy.optimize.minimize.
        Доступные варианты: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr'
    loss_type: str, optional
        Тип целевой функции
    tolerance: float, optional
        tolerance для scipy.optimize.minimize
    '''
    # минимизация
    result = sp.optimize.minimize(fun = Loss_direct,
                                x0 = start,
                                args = (loss_type, function_type, data),
                                method = minimization_method,
                                bounds = boundaries, 
                                tol = tolerance
                                )
    # возвращается модель
    return result       


def inverse_N_problems_solver(function_type: str,
                             data: np.ndarray,
                             start: list,
                             boundaries: list,
                             minimization_method: str ='COBYLA',
                             loss_type: str ='RMSE',
                             tolerance: float =1e-5,
                             ):
    '''Возвращает N моделей в виде объекта класса scipy.optimize.OptimizeResult и индекс модели с минимальной ошибкой
    
    Parameters
    ----------
    function_type: str
        Тип минимизируемой функции: 'rhoa' - кажущееся сопротивление, 'u' - разность потенциалов, 'E' - электрическое поле    
    data: numpy.ndarray
        Массив формы (K,2), K = количество моделей, data[i]=[r_i,f_i], r_i - полуразнос, f_i -измеренное значение
    start: list
        Список списков из стартовых значений для минимизации loss
    boundaries: list
        Список списков из кортежей границ значений параметров среды
    minimization_method: str, optional
        Метод оптимизации для scipy.optimize.minimize. \n
        Доступные варианты: 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr'
    loss_type: str, optional
        Тип целевой функции
    tolerance: float, optional
        tolerance для scipy.optimize.minimize
    '''

    results = []
    results_losses = []

    for i in range(len(data)):
        results.append(inverse_problem_solver(function_type,
                                              minimization_method, 
                                              data[i], 
                                              minimization_method=minimization_method, 
                                              loss_type=loss_type, 
                                              start=start[i],
                                              boundaries=boundaries[i],
                                              tolerance = tolerance,
                                              ))

    # возвращается список моделей и номер с минимальным значением ошибки loss_type
    return results, np.where(results_losses == np.min(results_losses))[0][0]

if __name__  != '__main__':
    print('inverse_problem was imported')
