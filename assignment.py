import numpy as np


def Dielectric_const(T,f):
    if T > 273.15:
        theta = 300 / T
        eta0 = 77.66 + 103.3 * (theta - 1)
        eta1 = 0.0671 * eta0
        eta2 = 3.52
        gamma1 = 20.2 - 146 * (theta - 1) + 316 * pow((theta - 1), 2)
        gamma2 = 39.8 * gamma1
        eta = eta0 - f * ((eta0 - eta1) / (f + 1j * gamma1) + (eta1 - eta2) / (f + gamma2 * 1j))
        eta = eta.conjugate()
    else:
        eta = 3.15 - 0.01 * 1j
    
    return eta
    
def Reflection_indices(dconst, angle):
    '''
    angle -- degrees
    '''
    n2 = np.sqrt(dconst)
    theta = np.radians(angle)
    p = np.sqrt(pow(n2, 2) - pow(np.sin(theta), 2))
    V = (np.cos(theta) - p) / (np.cos(theta) + p)
    H = (p - pow(n2, 2) * np.cos(theta)) / (p + pow(n2, 2) * np.cos(theta))
    V = np.power(np.abs(V), 2)
    H = np.power(np.abs(H), 2)
    return V, H
    
a = Dielectric_const(290, 30)

v, h = Reflection_indices(a, 30)