import numpy as np


def Dielectric_const(T,f):
    
    if T > 273.15:
        theta = 300 / T
        eta0 = 77.66 + 103.3 * (theta - 1.0)
        eta1 = 0.0671 * eta0
        eta2 = 3.52
        gamma1 = 20.2 - 146 * (theta - 1) + 316 * pow((theta - 1), 2)
        gamma2 = 39.8 * gamma1
        eta = eta0 - f * ((eta0 - eta1) / (f + 1j * gamma1) + (eta1 - eta2) / (f + gamma2 * 1j))
        eta = eta.conjugate()
    else:
        if hasattr(f, "__len__"):
            eta = np.ones(len(f)) * (3.15 - 0.01 * 1j)
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
    
def trans_bright(f,theta, temp, waterc):
    T0 = 255
    theta = np.radians(theta)
    gamma = waterc * f * 0.6
    tao = gamma / np.cos(theta)
    
    t = np.exp(-tao)
    T = T0 * t + temp * (1 - t)
    return t, T

def seaice_fm(f, polar, theta, ice_temp, ice_frac, waterc, temp):
    num = len(f)
    t, T = trans_bright(f, theta, temp, waterc)
    eta = Dielectric_const(ice_temp, f)
    V, H = Reflection_indices(eta, theta)
    return V,H
    
#t, T = trans_bright(30.0, 30.0, 290.0, 0.01)    
f = np.linspace(20, 90, 15)
V, H = seaice_fm(f, 0, 30, 260, 0, 0.01, 290) 

    
#a = Dielectric_const(260.0, 30)
#v, h = Reflection_indices(a, 30.0)