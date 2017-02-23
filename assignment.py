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
    
def trans_bright(f, theta, temp, waterc, T0 = 255):
    '''
    model of surface obsevations
    '''
    theta = np.radians(theta)
    gamma = waterc * f * 0.6
    tao = gamma / np.cos(theta)
    # TRANSMITIVITY
    t = np.exp(-tao)
    T = T0 * t + temp * (1 - t)
    return t, T
    

def seaice_fm(f, polar, theta, ice_temp, ice_frac, waterc, temp):
    # WATER TEMPERTURE ASSUMPTION 
    water_temp = 277.15
    eta_water = Dielectric_const(water_temp, f)
    eta_ice = Dielectric_const(ice_temp, f)
    Vw, Hw = Reflection_indices(eta_water, theta)
    ref_water = np.column_stack((Vw, Hw))
    Vi, Hi = Reflection_indices(eta_ice, theta)
    ref_ice = np.column_stack((Vi, Hi))
    #  POLORIZATION 
    po_water = []
    po_ice = []
    for i, j in enumerate(polar):
        po_water.append(ref_water[i][j])
        po_ice.append(ref_ice[i][j])
    po_water = np.array(po_water)
    po_ice = np.array(po_ice)
#    # ATMOSPHERIC EMISSION
#    t, T = trans_bright(f, theta, temp, waterc)
    # SURFACE EMISSION
    ti, Ti = trans_bright(f, theta, temp, waterc, ice_temp)
    tw, Tw = trans_bright(f, theta, temp, waterc, water_temp)
    Tb = Ti * ice_frac * (1 - po_ice) + Tw * (1 - ice_frac) * (1 - po_water)
    
    return Tb


f = np.array([19.7, 19.7, 37, 37, 85.5, 85.5])
polar = np.array([1, 0, 1, 0, 1, 0])

Tb = seaice_fm(f, polar, 30, 260, 0.5, 0.0, 290)

#t, T = trans_bright(30.0, 30.0, 290.0, 0.01)       
#a = Dielectric_const(290.0, 30.0)
#v, h = Reflection_indices(a, 30.0)