import numpy as np
from GeneralFunction import *
from Ship_FCM import *
import matplotlib.pyplot as plt

class state:
    def __init__(self, longi, lati, bearing):
        self.longi = longi
        self.lati = lati
        self.bearing = bearing
        self.time = 0.0
        self.speed = 0.0
        
def cultime(initial_time, delta_t):
    num = len(delta_t)
    t = []
    t.append(initial_time)
    for i in range(num):
        t.append(initial_time + delta_t[:i+1].sum())
    return np.array(t)
        
        

# departure
p_dep = np.array([-5.0, 49.0])
# destination
p_des = np.array([-65.0, 40.0])

X,Y = m.gcpoints(p_dep[0], p_dep[1], p_des[0], p_des[1], 40)

x, y = m(X,Y, inverse = True)
dist_bearing = []

for i in range(39):
    dist_bearing.append(greatcircle_inverse(x[i], y[i], x[i+1], y[i+1]))
    
dist_bearing = np.array(dist_bearing)    
    
info = np.column_stack((x[:-1], y[:-1], dist_bearing))    
    
    
initial_time = 0
velocity = 20
delta_t = []
for i in range(39):
    delta_t.append(info[i,2] / velocity / 1.852)
delta_t = np.array(delta_t)
ti = cultime(initial_time, delta_t)

total_time = sum(delta_t) + initial_time

    
    


#plt.figure(figsize=(20, 15))
#
#m.drawgreatcircle(p_dep[0], p_dep[1], p_des[0], p_des[1], linewidth=2,color='b')
#m.scatter(X, Y, marker='D',color='g')
#m.drawcoastlines()
#m.fillcontinents()
#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#plt.show()