import numpy as np


def geological_environment(params):
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