import numpy as np


def geological_environment(params):
    '''Принимает list параметров среды и возвращает прямоугольный np.array сопротивлений в среде

    Parameters
    ----------
    param: numpy.ndarray
      Массив параметров среды формой (2N-1), ultN -количество слоёв в модели. param[2*(i-1)]=rhoa_i, i=1, ..., N; param[2*(i-1)+1] = thickness_i, i=1, ..., N-1
    '''
    
    
    matrix=params[0]*np.ones((int(sum(params[1::2])*1.25), int(sum(params[1::2])*1.25)))
    for i in range(matrix.shape[0]):
        for k in range(0,int(len(params)/2+1/2)):
            if sum(params[1:2*k:2])<i<=sum(params[1:2*(k+1):2]):
                matrix[i][:]=params[2*k]
        if sum(params[1:2*k:2])<i:
            matrix[i][:]=params[2*k]
    return matrix

if __name__  != '__main__':
    print('visualization_functions were imported')