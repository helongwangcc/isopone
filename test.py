import numpy as np
#from matplotlib import pyplot as plt
from scipy.integrate import simps

def jonswap(hs, tp, gamma):
    omega = np.linspace(0.1, np.pi / 2, 50)
    wp = 2 * np.pi / tp
    sigma = np.where(omega < wp, 0.07, 0.09)
    a = np.exp(-0.5 * np.power((omega - wp) / (sigma * omega), 2.0))
    sj = 320 * np.power(hs, 2) * np.power(omega, -5.0) / np.power(tp, 4) * np.exp(-1950 * np.power(omega, -4) / np.power(tp, 4)) * np.power(gamma, a)
    amp1 = simps(sj,omega)
    di = np.diff(omega)
    amp2 = sj[:-1].T.dot(di)
#    amp = np.sqrt(2 * simps(sj, omega))
#    ind = np.argmax(sj)
#    wavelength = 2 * np.pi * 9.81 / np.power(omega[ind], 2)
#    return sj, amp
    return amp1,amp2
    

hs1 = 5
tp1 = 11
gamma1 = 5.5

sw1 = jonswap(hs1,tp1,gamma1)

#gamma2 = 5.0
#sw2 = jonswap(ome, hs1,tp1,gamma2)
#
#plt.figure(figsize=(10, 6))
#
#plt.plot(ome, sw1, 'k')
#plt.plot(ome, sw2, 'r')
#plt.show()


