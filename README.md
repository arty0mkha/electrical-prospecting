# Electrical-prospecting
GGD NSU 21501



Структура:
	
	I) Папка с кодом (файлы в формате .py):
		1) блок с решением прямой задачи
		2) блок с решением обратной задачи:
			2.1 inverse_problem() - решение обратной задачи с использованием direct_problem()
			2.2 aprox_inverse_problem() решение обратной задачи через использование двухслойной модели	
		3) блок с анализом direct_problem() - чувствительность и эквивалентность
  	II) Папка с данными
	III) Файл с визуализацией (визуализация всего)

Необоходимые библиотеки:
	scipy
	numpy
	matplotlib




Общие переменные:	

	param: numpy.ndarray
        Массив параметров среды формой (2N-1), N -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1  
 	loss_type: str, optional
        Тип целевой функции         
	data: numpy.ndarray
        Массив формы (K,2), K = количество измерений, data[i]=[r_i,f_i], r_i - полуразнос, f_i -измеренное значение       

     
