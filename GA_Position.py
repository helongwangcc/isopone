import numpy as np
from numpy.random import uniform 
from numpy.random import rand, randint, normal
from GeneralFunction import *
from Ship_FCM import *
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime, timedelta


# BATHYMETRIC CLASS
bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")
# WATER DEPTH LIMITATION (m)
DEPTH_LIMIT = -11
# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
# INITIATE SHIP INFORMATION
ship_info = Ship_FCM()

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
        # SELECT POINTS USING NORMAL DISTRIBUTION 
        mean_node = (self.num - 1) / 2
        std_node = float(self.num) / 6
        node_ind = []
        while True:
            if len(node_ind) >= length_set - 2:
                break
            index = int(normal(mean_node, std_node) + 0.5)
            if (index >= 0) & (index < self.num):
                node_ind.append(index)
        node_ind.insert(0, 0)
        node_ind.append(0)
        node_ind = np.array(node_ind)
        # SELECT POINTS USING UNIFORM DISTRIBUTION
#        node_ind = randint(0, self.num - 1, length_set - 2)
#        points = []
#        points.append(self.nodeset[0])
#        for i, j in enumerate(self.nodeset[1:-1]):
#            points.append(j[node_ind[i]])
#        points.append(self.nodeset[-1])
        rate_ind = randint(0, length_rate, length_set - 1)
            
        
        return [node_ind, rate_ind]
    
    def population(self, count):
        '''
        CREATE A NUMBER OF INDIVIDUALS
        '''
        container = []
        while len(container) < count:
            # ADD CONSTRAINTS FOR EACH INDIVIDUALS
            individual = self.individual()
            ind = []
            pos = individual[0]
            for i in range(len(pos) - 1):
                if i == 0:
                    ind.append(bathymetry.is_below_depth(self.nodeset[i], self.nodeset[i+1][pos[i+1]], DEPTH_LIMIT))
                elif i == len(pos) - 2:
                    ind.append(bathymetry.is_below_depth(self.nodeset[i][pos[i]], self.nodeset[i+1], DEPTH_LIMIT))
                else:
                    ind.append(bathymetry.is_below_depth(self.nodeset[i][pos[i]], self.nodeset[i+1][pos[i+1]], DEPTH_LIMIT))
            ind = np.array(ind)
            if np.all(ind) == False:
                continue
            else:
                container.append(individual)
        
        return container
    
    def fitness(self, individual, Initial_time, speed):
        '''
        HYPOTHESIS AND LEARNING
        '''
        # HEADING
        Posin, Pin = individual
        Pos = []
        for i in range(len(Posin)):
            if i == 0:
                Pos.append(self.nodeset[i])
            elif i == len(Posin) - 1:
                Pos.append(self.nodeset[i])
            else:
                Pos.append(self.nodeset[i][Posin[i]])
        Pos = np.array(Pos)
        Heading = []
        Dist = []
        fuel = []
        for i in range(len(Pos) - 1):
            Dist.append(greatcircle_inverse(Pos[i][0], Pos[i][1], Pos[i+1][0], Pos[i+1][1])[0])
            Heading.append(greatcircle_inverse(Pos[i][0], Pos[i][1], Pos[i+1][0], Pos[i+1][1])[1])
        Heading = np.array(Heading)
        Dist = np.array(Dist)
        Powerrate = self.powerrate[Pin]
        time = Initial_time
        for i in range(len(Pos) - 1):
            Hs = weather_info.hs([Pos[i][1], Pos[i][0], time])        
            wind_U = weather_info.u([Pos[i][1], Pos[i][0], time]) / 0.5144
            wind_V = weather_info.v([Pos[i][1], Pos[i][0], time]) / 0.5144        
            Tp = weather_info.tp([Pos[i][1], Pos[i][0], time])
            head = weather_info.hdg([Pos[i][1], Pos[i][0], time])
            cu = weather_info.cu([Pos[i][1], Pos[i][0], time])
            cv = weather_info.cv([Pos[i][1], Pos[i][0], time])
            # WATER DEPTH
            depth = bathymetry.water_depth(Pos[i])
            v = ship_info.Power_to_speed(speed, Powerrate[i], Heading[i], cu, cv, depth, wind_U, wind_V, Hs, Tp, head)
            fuelc = ship_info.weather2fuel(v, Heading[i], cu, cv, depth, wind_U, wind_V, Hs, Tp, head)
            fuel.append(fuelc[3] * Dist[i] / 1.852)
            time = Dist[i] / v / 1.852 + time
            
        
        return np.sum(fuel), time
    
    def Pareto_search(self, Initial_time, speed):
        #INITIAL DISTANCE
        dist = 20
        container = []
        while len(container) < dist:
            pop = self.population(1000)
            if container ==[]:
                pass
            else:
                for element in container:
                    pop.append(element)
            results = []
            for individual in pop:
                results.append(self.fitness(individual, Initial_time, speed))
            results = np.array(results)
            is_efficient = np.ones(results.shape[0], dtype = bool)
            for i, c in enumerate(results):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(results[is_efficient] <= c, axis = 1)
            container = []
            for i in range(len(pop)):
                if is_efficient[i] == True:
                    container.append(pop[i])
            graded = [(results[i][0], pop[i]) for i in xrange(len(pop))]
      
        return container, results[is_efficient]
            
        
        
    def evolve(self, pop, Initial_time, speed, retain = 0.2, random_select = 0.05, mutate = 0.01):
        parents = []
        results = []
        for individual in pop:
            results.append(self.fitness(individual, Initial_time, speed))
        results = np.array(results)
        is_efficient = np.ones(results.shape[0], dtype = bool)
        for i, c in enumerate(results):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(results[is_efficient] <= c, axis = 1)
        
        print results[is_efficient]
        graded = [(results[i][0], pop[i]) for i in xrange(len(pop))]
        graded = [x[1] for x in sorted(graded, key = lambda tup : tup[0])]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]

        # RANDOMLY ADD OTHER INDIVIDUALS TO PROMOTE GENETIC DIVERSITY
        for individual in graded[retain_length:]:
            if random_select > rand():
                parents.append(individual)  
            
        # MUTATE SOME INDIVIDUALS
        # SELECT POINTS USING NORMAL DISTRIBUTION 
        mean_node = (self.num - 1) / 2
        std_node = float(self.num) / 6
        for individual in parents:
            if mutate > rand():
                pos_to_mutate = randint(0, len(individual[0]) - 1)
                index = int(normal(mean_node, std_node) + 0.5)
                if (index >= 0) & (index < self.num):
                    individual[0][pos_to_mutate] = index
                individual[1][pos_to_mutate] =  randint(0, len(self.powerrate))                
                
        # ADD PARETO SOLUTIONS        
        for i in range(len(pop)):
            if pop[i] in parents:
                print i
#            if is_efficient[i] == True:
#                if not pop[i] in parents:
#                    parents.append(pop[i])
        
        
        # CROSSOVER PARENTS TO CREATE CHILDREN
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length - 1)
            female = randint(0, parents_length - 1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = len(male) / 2
                child = male[:half] + female[half:]
                children.append(child)
    
        parents.extend(children)
        
        return parents 
        
#    def Pareto_solution(pop)
                         







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

ge = GA_position(p_dep, p_des, 20, 6, 51, 0.2)

pop = ge.population(100)
for i in range(20):
    pop = ge.evolve(pop, 0, 20)



#s = ge.individual()
#res = ge.fitness(s,0,20)
#pop = ge.population(100)
#population, res = ge.Pareto_search(0, 20)
#info = res[ind]
#p = []
#for i,j in enumerate(pop):
#    if np.any(info[:,2] == i):
#        p.append(pop[i][0])
#
#
##grid_drawing(ge.nodeset)
#m = Basemap(
#  projection="merc",
#  resolution='l',
#  area_thresh=0.1,
#  llcrnrlon=-75,
#  llcrnrlat=35,
#  urcrnrlon=10,
#  urcrnrlat=55
#)
##
##
#plt.figure(figsize=(20, 15))
#grid = ge.nodeset
#for i in range(len(grid)):
#    
#    if i == 0:
#        x, y = grid[i]
#        X, Y = m(x, y)
#        m.scatter(X, Y, marker='D',color='r')
#    elif i == len(grid) - 1:
#        x, y = grid[i]
#        X, Y = m(x, y)
#        m.scatter(X, Y, marker='D',color='k')
#    else:
#        x = grid[i][:,0]
#        y = grid[i][:,1]
#        X, Y = m(x, y)
#        m.scatter(X, Y, marker='*',color='g')
#for _ in p:
#    x, y = m(_[:,0],_[:,1])
#    m.plot(x,y,color='g')
#m.drawcoastlines()
#m.fillcontinents()
#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#plt.show()
