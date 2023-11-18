# electrical-prospecting
GGD NSU 21501



Структура:
1) функция direct_problem() - решает прямую задачу
2) блок с решением обратной задачи:
   
  	2.1 inverse_problem() - решение обратной задачи с использованием direct_problem()
  
  	2.2 approx_inverse_problem() решение обратной задачи через использование двухслойной модели
  
4) блок с анализом direct_problem() - чувствительность и эквивалентность
5) блок с отрисовкой всего этого


Переменные:

    param: numpy.ndarray
        Массив параметров среды формой (2N-1), N -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1  
				
    loss_type: str
        Тип целевой функции   
    
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

     
