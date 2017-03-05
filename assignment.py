import numpy as np
from numpy.random import normal
from numpy.linalg import lstsq



def Dielectric_const(T, f):
    
    if T > 272.25:
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
    
def trans_bright(f, theta, temp, waterc):
    '''
    model of surface obsevations
    '''
    theta = np.radians(theta)
    gamma = waterc * f * 0.6
    tao = gamma / np.cos(theta)
    # TRANSMITIVITY
    t = np.exp(-tao)
    T = temp * t 
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
    # ATMOSPHERIC EMISSION
    t, T = trans_bright(f, theta, temp, waterc)    
    # SURFACE EMISSION    
    Tb = ice_temp * (1 - po_ice) * ice_frac * t + water_temp * (1 - po_water) * (1 - ice_frac) * t + temp * (1 - t) * t * ((1 - ice_frac) * po_water + ice_frac * po_ice) + temp * (1 - t) 
    
    return Tb

def method1(f, polar, theta, Tb):
    '''
    inverse process
    '''
    water_temp = 273.15
    eta_water = Dielectric_const(water_temp, f)
    eta_ice = Dielectric_const(260, f)
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
    K = np.column_stack((-(1 - po_water) * water_temp,  1 - po_ice))
    y = Tb - (1 - po_water) * water_temp
    
    return K, y

def method2(f, polar, theta, ice_temp, ice_frac, temp, Tb):
        # WATER TEMPERTURE ASSUMPTION 
    water_temp = 273.15
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
    # ATMOSPHERIC EMISSION
    t, T = trans_bright(f, theta, temp, 0)    
    delta1 = t * ((1 - po_ice)  * ice_temp - (1 - po_water) * water_temp + temp * (1 - t) * (po_ice - po_water))
    delta2 = ice_frac * (1 - po_ice)
    K = np.column_stack((delta1, delta2))
    y = Tb - seaice_fm(f, polar, theta, ice_temp, ice_frac, 0, temp)
    return K, y

    
def method2_iter(f, polar, theta, ice_temp, ice_frac, temp, Tb):
    
    K, y = method2(f, polar, theta, ice_temp, ice_frac, temp, Tb)
    ans = lstsq(K, y)[0]
    while ans[0] > 10e-3:
        ice_frac = ice_frac + ans[0]
        K, y = method2(f, polar, theta, ice_temp, ice_frac, temp, Tb)
        ans = lstsq(K, y)[0]
    while ans[1] > 10e-2:
        ice_temp = ice_temp + ans[1]
        K, y = method2(f, polar, theta, ice_temp, ice_frac, temp, Tb)
        ans = lstsq(K, y)[0]
    
    return ice_frac, ice_temp

# FREQUECY SETTING
f = np.array([19.7, 19.7, 37, 37, 85.5, 85.5])
# POLARIZATION SETTING
polar = np.array([1, 0, 1, 0, 1, 0])
# INCIDENCE ANGLE
theta = 30.0
# ICE TEMPERTURE
ice_temp = 260.0
# ICE FRACTION
ice_frac = 0.5
# WATER CONTENT
waterc = 0
# CLOUD TEMPERTURE
temp = 270
# BRIGHTNESS TEMPERTURE 
Tb = seaice_fm(f, polar, theta, ice_temp, ice_frac, waterc, temp)
# METHOD A ADDING NOIZE
Tb1 = Tb + normal(0, 0.5, len(Tb))

###############################################################################

## GENERATE MATRIX EQUATION METHOD 1
#K1, y1 = method1(f, polar, theta, Tb1)
## LEAST SQUARES FITS
#Ans1 = lstsq(K1, y1)[0]

###############################################################################

## INITIAL GUESS
#ice_temp1 = 240
#ice_frac1 = 0.3
## GENERATE MATRIX EQUATION METHOD 2
#ice_temp1, ice_frac1 = method2_iter(f, polar, theta, ice_temp1, ice_frac1, temp, Tb1)

############################################################################### 

def methodB(f, polar, theta, ice_temp, ice_frac, waterc, temp, Tb):
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
    # ATMOSPHERIC EMISSION
    t, T = trans_bright(f, theta, temp, waterc)    
    delta1 = t * ((1 - po_ice)  * ice_temp - (1 - po_water) * water_temp + temp * (1 - t) * (po_ice - po_water))
    delta2 = ice_frac * (1 - po_ice)
    delta3 = - (f * 0.6 / np.cos(np.radians(theta))) * t *(ice_frac * (1 - po_ice) * ice_temp + (1 - ice_frac) *(1 - po_water) * water_temp + temp * (-1 + (1 - 2 * t) * (ice_frac * po_ice + (1 - ice_frac) * po_water)))
    K = np.column_stack((delta1, delta2, delta3))
    y = Tb - seaice_fm(f, polar, theta, ice_temp, ice_frac, waterc, temp)
    return K, y
    
   
def fitness(f, polar, theta, ice_temp, ice_frac, waterc, temp, Tb):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    k, y = methodB(f, polar, theta, ice_temp, ice_frac, waterc, temp, Tb)
    s = lstsq(k,y)[0]
    return s

#k, y = methodB(f, polar, theta, ice_temp, ice_frac, waterc, temp, Tb1)



# INITIAL GUESS
ice_temp1 = 240
ice_frac1 = 0.3
waterc1 = 0.03

s = fitness(f, polar, theta, ice_temp, ice_frac, waterc1, temp, Tb)