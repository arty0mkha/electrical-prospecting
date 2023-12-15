# Electrical-prospecting
GGD NSU 21501



Структура:
	
	I) Папка с кодом (файлы в формате .py):
		1) блок с решением прямой задачи
		2) inverse_problem - решение обратной задачи
		3) sensivity - чувствительность и эквивалентность
		4) visualization_functions - набор вспомагательных функций для визуализации
  	II) Папка с данными
	III) _example - файлы  с примерами работы с кодом
	IV) visualization_ - файлы-решения задач с визуализацией

Необоходимые библиотеки:

		scipy
		numpy
		matplotlib
		tqdm



Общие переменные:	

	param: list
        Список параметров среды формой (2N-1), N -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1  
 	loss_type: str, optional
        Тип целевой функции         
	data: numpy.ndarray
        Массив формы (K,2), K = количество измерений, data[i]=[r_i,f_i], r_i - полуразнос, f_i -измеренное значение       

     
