import numpy as np
from numpy.random import uniform 
from numpy.random import rand, randint
from GeneralFunction import *
from Ship_FCM import *
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime, timedelta




class GA_position:
    
    def __init__(self, departure, destination, v, delta_t, n, eta):
        '''
        DYNAMIC PROGRAMMING GRID SYSTEM IS USED 
        NO CLASS CONTAINER USED
        '''
        # TOTAL DISTANCE AND INITIAL BEARING
        Total_dist, Init_bearing = greatcircle_inverse(departure[0], departure[1], destination[0], destination[1])
        # K = STAGE NUMBER
        K = np.int(np.ceil(Total_dist / (v * 1.854 * delta_t)))
        # UNIT SPACING
        delta_y = Total_dist * eta / (n - 1)
        x, y = m.gcpoints(departure[0], departure[1], destination[0], destination[1], K)
        longi_c, lati_c = np.array(m(x, y, inverse = True))
        # DISTANCE BETWEEN EACH STAGES
        para_stage = greatcircle_inverse(longi_c[:-1], lati_c[:-1], longi_c[1:], lati_c[1:]).T
        self.nodeset = []
        self.nodeset.append(np.array([longi_c[0], lati_c[0]]))
        self.num = n
        for i in range(1, K - 1):
            temp_brng = greatcircle_point(longi_c[i - 1], lati_c[i - 1], para_stage[i - 1, 0], para_stage[i - 1, 1])[2]
            temp_point1 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng + 90)
            temp_point2 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng - 90)
            temp_x, temp_y = m.gcpoints(temp_point1[0], temp_point1[1], temp_point2[0], temp_point2[1], n)
            temp_pos =np.array(m(temp_x, temp_y ,inverse = True)).T
            # NODE NUMBER START AT 0
            self.nodeset.append(np.array([tempnode for tempnode in temp_pos]))
        self.nodeset.append(np.array([longi_c[-1], lati_c[-1]]))
        ####
        self.powerrate = np.array([0.5, 0.6, 0.75, 0.85, 1.0])
        
    def individual(self):
        '''
        CREATE A MEMBER OF POPULATION 
        '''
        length_set = len(self.nodeset)
        length_rate = len(self.powerrate)
        
        node_ind = randint(0, self.num - 1, length_set - 2)
        rate_ind = randint(0, length_rate - 1, length_set - 1)
        points = []
        points.append(self.nodeset[0])
        for i, j in enumerate(self.nodeset[1:-1]):
            points.append(j[node_ind[i]])
        points.append(self.nodeset[-1])
            
        
        return np.array(points)
        
        
        







def grid_drawing(grid):
    '''
    DRAWING GRID 
    PLEASE CHECK THE UPPER AND LOWER BOUNDARYS OF THE MAP
    '''
    # MAP INITIALIZATION
    m = Basemap(
      projection="merc",
      resolution='l',
      area_thresh=0.1,
      llcrnrlon=-75,
      llcrnrlat=35,
      urcrnrlon=10,
      urcrnrlat=55
    )
    ####
    plt.figure(figsize=(20, 15))
    for i in range(len(grid)):
        if i == 0:
            x, y = grid[i]
            X, Y = m(x, y)
            m.scatter(X, Y, marker='D',color='r')
        elif i == len(grid) - 1:
            x, y = grid[i]
            X, Y = m(x, y)
            m.scatter(X, Y, marker='D',color='k')
        else:
            x = grid[i][:,0]
            y = grid[i][:,1]
            X, Y = m(x, y)
            m.scatter(X, Y, marker='*',color='g')
    m.drawcoastlines()
    m.fillcontinents()
    m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
    m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#    plt.show()
            
    
# departure
p_dep = np.array([-5.0, 49.0])
# destination
p_des = np.array([-65.0, 40.0])

ge = GA_position(p_dep, p_des, 20, 6, 15, 0.2)

s = ge.individual()

#grid_drawing(ge.nodeset)
m = Basemap(
  projection="merc",
  resolution='l',
  area_thresh=0.1,
  llcrnrlon=-75,
  llcrnrlat=35,
  urcrnrlon=10,
  urcrnrlat=55
)


plt.figure(figsize=(20, 15))
grid = ge.nodeset
for i in range(len(grid)):
    
    if i == 0:
        x, y = grid[i]
        X, Y = m(x, y)
        m.scatter(X, Y, marker='D',color='r')
    elif i == len(grid) - 1:
        x, y = grid[i]
        X, Y = m(x, y)
        m.scatter(X, Y, marker='D',color='k')
    else:
        x = grid[i][:,0]
        y = grid[i][:,1]
        X, Y = m(x, y)
        m.scatter(X, Y, marker='*',color='g')
x, y = m(s[:,0],s[:,1])
m.plot(x,y,color='g')
m.drawcoastlines()
m.fillcontinents()
m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
plt.show()
