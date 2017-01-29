import numpy as np
from numpy.random import uniform 
from numpy.random import rand, randint
from GeneralFunction import *
from Ship_FCM import *
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime, timedelta

# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
# INITIATE SHIP INFORMATION
ship_info = Ship_FCM()
# BATHYMETRIC CLASS
bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")

class state:
    def __init__(self, dep, des, n, initial_time, service_speed):
        self.initial_time = initial_time
        X,Y = m.gcpoints(dep[0], dep[1], des[0], des[1], n)
        self.x, self.y = m(X,Y, inverse = True)
        dist_bearing = []
        for i in range(n - 1):
            dist_bearing.append(greatcircle_inverse(self.x[i], self.y[i], self.x[i+1], self.y[i+1]))
        dist_bearing = np.array(dist_bearing)    
        self.info = np.column_stack((self.x[:-1], self.y[:-1], dist_bearing)) 
        self.delta_t = self.info[:,2] / service_speed / 1.852
        self.container = []
    
    def cultime(self, delta_new_t):
        num = len(delta_new_t)
        t = []
        t.append(self.initial_time)
        for i in range(num):
            t.append(self.initial_time + delta_new_t[:i+1].sum())
        return np.array(t)
        
    def individual(self):
        '''
        Create a member of population
        '''
        return uniform(-min(self.delta_t) / 6, max(self.delta_t), len(self.delta_t))
    
    def population(self, count):
        '''
        Create a number of individuals
        '''
        container = []
        i = 0
        time_limit = 200
        while i < count:
            individual = self.individual()
            if max(self.cultime(self.delta_t + individual)) > time_limit:
                continue
            else:
                container.append(individual)
                i = i + 1
            
        return np.array(container)
    
    def ancestor(self, count):
        '''
        Create first generation
        '''
        # SET CALCULATION 
        i = 0
        container = []    
        while True:
            if (i > 100000) & (container == []):
                print "cannot find proper combination"
                break
            individual = self.individual()
            delta_tn = self.delta_t +individual
            ti = self.cultime(delta_tn)
            stat = np.column_stack((self.info[:,1], self.info[:,0], ti[:-1]))
            Hs = weather_info.hs(stat)
            if np.any(Hs > 7) == True:
                i = i + 1
                continue
            else:
                container.append(individual)
                i = i + 1     
                
        return container       
        
    def fitness(self, individual, target):
        delta_tn = self.delta_t + individual
        ti = self.cultime(delta_tn)
        # STATE FOR EACH WAYPOINTS
        stat = np.column_stack((self.info[:,1], self.info[:,0], ti[:-1]))
        Hs = weather_info.hs(stat)        
        wind_U = weather_info.u(stat) / 0.5144
        wind_V = weather_info.v(stat) / 0.5144        
        Tp = weather_info.tp(stat)
        head = weather_info.hdg(stat)
        cu = weather_info.cu(stat)
        cv = weather_info.cv(stat)
        # WATER DEPTH
        depth = np.array([bathymetry.water_depth(p) for p in self.info[:,:2]]).ravel()
        vn = self.info[:,2] / delta_tn / 1.852
        # FUEL CONSUMPTION; OUTPUT :[PE, PS, FUEL/H, FUEL/NM]self.individual() for x in xrange(count)
        params = zip(vn, self.info[:,3], cu, cv, depth, wind_U, wind_V, Hs, Tp, head)
        power = np.array([ship_info.weather2fuel(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]) for p in params])
        fuelc = sum(self.info[:,2] / 1.852 * power[:,3])
    
        return abs(fuelc - target)
    
    def grade(self, pop, target):
        summed = sum([self.fitness(x, target) for x in pop])
        return summed / (len(pop) * 1.0)
    
    def evolve(self, pop, target, retain = 0.2, random_select = 0.05, mutate = 0.01):
        graded = [(self.fitness(x, target), x) for x in pop]
        graded = [x[1] for x in sorted(graded, key = lambda tup : tup[0])]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]
    
        # RANDOMLY ADD OTHER INDIVIDUALS TO PROMOTE GENETIC DIVERSITY
        for individual in graded[retain_length:]:
            if random_select > rand():
                parents.append(individual)
    
    
        # MUTATE SOME INDIVIDUALS
        for individual in parents:
            if mutate > rand():
                changenum = randint(0, int(len(individual) / 5))
                pos_to_mutate = randint(0, len(individual) - 1, changenum)
                individual[pos_to_mutate] = uniform(min(individual), max(individual), changenum)
    
    
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
                child = male[:half].tolist() + female[half:].tolist()
                child = np.array(child)
                children.append(child)
    
        parents.extend(children)
    
    
        return np.array(parents)
    
    def recursion(self, target, num_population, num_recursion, retain = 0.2, random_select = 0.05, mutate = 0.01):
        self.container = []
        p = self.population(num_population)
        fitness_history = [self.grade(p,target)]
        for i in xrange(num_recursion):
            self.container.append(p)
            p = self.evolve(p, target, retain, random_select, mutate)
            fitness_history.append(self.grade(p, target))
        
        return p, fitness_history
    
    def gatherinfo(self, pop, target):
        graded = sorted([(self.fitness(x, target), x) for x in pop], key = lambda tup : tup[0])
        delta_t = self.delta_t + graded[0][1]
        ti = self.cultime(delta_t)
        speed = (self.info[:, 2] / delta_t / 1.852).tolist()[::-1]
        speed.append(0.0)
        speed = np.array(speed[::-1])
        ginfo = np.column_stack((self.x, self.y, ti, speed))
        
        return ginfo
       
## departure
#p_dep = np.array([-5.0, 49.0])
## destination
#p_des = np.array([-65.0, 40.0])
#
## lowest fuel consumption
#calm_r = ship_info.weather2fuel(20, 0, 0, 0, -10000, 0, 0, 0.01, 0.01, 0)
#minifuelc = calm_r[3] * greatcircle_inverse(p_dep[0], p_dep[1], p_des[0], p_des[1])[0] / 1.852
#
#
#initial_time = 25
#velocity = 20
#
#
#gentic = state(p_dep,p_des, 40, initial_time, velocity)
#
#
#pop, results = gentic.recursion(0, 1000, 5, 0.2, 0.05, 0.1)
#inform =gentic.gatherinfo(pop, minifuelc)







#plt.figure(figsize=(20, 15))
#
#m.drawgreatcircle(p_dep[0], p_dep[1], p_des[0], p_des[1], linewidth=2,color='b')
#m.scatter(X, Y, marker='D',color='g')
#m.drawcoastlines()
#m.fillcontinents()
#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#plt.show()




#########################################################################

#m = Basemap(
#  projection="merc",
#  resolution='l',
#  area_thresh=0.1,
#  llcrnrlon=-75,
#  llcrnrlat=35,
#  urcrnrlon=10,
#  urcrnrlat=55
#)





#############################################################
# animation 



#fig = plt.figure(figsize=(20, 15))
#lon, lat = np.meshgrid(weather_info.lon, weather_info.lat)
#x, y = m(lon, lat)
#
#x1, y1 = m(inform[:,0], inform[:,1])
#it = inform[:, 2]
#wt = np.linspace(0, (124 - 1) * 6, 124)
#ani_ind = int(np.argwhere(wt > max(it))[0])
#anj_ind = int(np.argwhere(wt < min(it))[-1])
#
#
#txt = plt.title('', fontsize = 20)
#
#
#m.drawcoastlines()
#m.fillcontinents()
#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#
#def init():
#    cs = m.pcolormesh(x, y, weather_info.Hs[:,:,0], shading='flat', cmap=plt.cm.jet)
#    md = weather_info.Met['Time'][0][0]
#    pd = datetime.fromordinal(int(md)) + timedelta(days=md%1) - timedelta(days = 366)
#    txt.set_text("GA @ Significant wave height at " + str(pd))
#    cb = plt.colorbar(cs, orientation ='horizontal', fraction=0.1,shrink=.5, pad=0.05)
#    cb.ax.tick_params(labelsize=20) 
#    return [cs]
#
#def animate(i):
#    #
#    if i < anj_ind+1:
#        cs = m.pcolormesh(x, y, weather_info.Hs[:,:,i], shading='flat', cmap=plt.cm.jet)
#        return [cs]
##    if i == 0:
##        cs = m.pcolormesh(x, y, weather_info.Hs[:,:,0], shading='flat', cmap=plt.cm.jet)
##        return [cs]
#    cs = m.pcolormesh(x, y, weather_info.Hs[:,:,i], shading='flat', cmap=plt.cm.jet)
#    index = int(max(np.argwhere(it<wt[i])))
#    m.plot(x1[:index + 1],y1[:index + 1] ,marker='o',markersize = 5, linewidth=3, color='k')
#    md = weather_info.Met['Time'][i][0]
#    pd = datetime.fromordinal(int(md)) + timedelta(days=md%1) - timedelta(days = 366)
#    txt.set_text("GA @ Significant wave height at " + str(pd))
#    return [cs]
#
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames= ani_ind+1, interval=1000, blit=True)
#mywriter = animation.FFMpegWriter(fps=1)
#
#anim.save('GA.mp4', writer=mywriter)



# color='#ff0080'
##########################################################################################
#hs = [weather_info.hs([i[1],i[0],i[2]]) for i in inform]
#plt.figure(figsize=(15, 10))
#plt.title("Hs During Voyage", fontsize = 20, fontweight='bold')
#plt.ylabel("Hs [m]", fontsize = 14)
#plt.xlabel("Longitude [Degree]", fontsize = 14)
#
#plt.plot(inform[:,0], hs, marker = "o", color = "g", label = "GA")
#
#
#plt.grid()
#plt.legend(loc = 1)
#plt.show()
