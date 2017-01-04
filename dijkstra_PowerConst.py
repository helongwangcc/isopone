import numpy as np
from GeneralFunction import *
from Ship_FCM import *
import heapq
from matplotlib import pyplot as plt
# BATHYMETRIC CLASS
bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")
# WATER DEPTH LIMITATION (m)
DEPTH_LIMIT = -11
# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
# INITIATE SHIP INFORMATION
ship_info = Ship_FCM()



class dnode:
    def __init__(self, longi, lati):
        self.longi = longi
        self.lati = lati

def gen_graph(departure, destination, v, delta_t, n, eta):
    '''
          v -- ship service speed in knot
    delta_t -- time interval between two stage
          n -- total number  of the grid points on one stage
        eta -- proportion between the breadth of one stage and Total distance
          q -- number of transition
    '''
    # total distance and intial bearing
    Total_dist, Init_bearing = greatcircle_inverse(departure[0], departure[1], destination[0], destination[1])
    # K = stage number
    K = np.int(np.ceil(Total_dist / (v * 1.854 * delta_t)))
    # unit spacing
    delta_y = Total_dist * eta / (n - 1)
    x, y = m.gcpoints(departure[0], departure[1], destination[0], destination[1], K)
    longi_c, lati_c = np.array(m(x, y, inverse = True))
    # distance between each stage
    para_stage = greatcircle_inverse(longi_c[:-1], lati_c[:-1], longi_c[1:], lati_c[1:]).T
    nodeset = []
    nodeset.append([dnode(longi_c[0], lati_c[0])])        
    for i in range(1, K - 1):
        temp_brng = greatcircle_point(longi_c[i - 1], lati_c[i - 1], para_stage[i - 1, 0], para_stage[i - 1, 1])[2]
        temp_point1 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng + 90)
        temp_point2 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng - 90)
        temp_x, temp_y = m.gcpoints(temp_point1[0], temp_point1[1], temp_point2[0], temp_point2[1], n)
        temp_pos =np.array(m(temp_x, temp_y ,inverse = True)).T
        # node number starts at 0
        nodeset.append([dnode(tempnode[0], tempnode[1]) for tempnode in temp_pos])
    nodeset.append([dnode(longi_c[-1], lati_c[-1])])
    
    return nodeset

   
class DijkstraGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = {}
    
    def in_bounds(self, id):
        (x, y) = id.T
        return np.all([x >= 0, x < self.width, y >= 0, y < self.height], axis = 0)
    
    def passable(self, id):
        is_wall = []
        for i in id:
            if i[1] in self.walls:
                is_wall.append(np.in1d(i[0], self.walls[i[1]]))
            else:
                is_wall.append([False])        
        is_wall = np.array(is_wall).ravel()
        return is_wall
    
    def neighbors(self, id, q, container):
        (x, y) = id # x: longitude, y: latitude
        if x == 0:
            interval = np.arange(len(container[1]))
            results = np.column_stack((np.ones(len(container[1])), interval)).astype(int)
        elif x == len(container) - 2:
            parent = container[x][y]
            child = container[x + 1][0]
            if bathymetry.is_below_depth((parent.longi, parent.lati),(child.longi, child.lati),DEPTH_LIMIT):
                return tuple(map(tuple, [[x + 1, 0]]))
            else:
                return
        else:
            interval = np.linspace(y - q, y + q, 2 * q + 1)
            results = np.column_stack((np.ones(2 * q + 1) * (x + 1), interval)).astype(int)
            results = results[self.in_bounds(results)]
        ind = []
        parent = container[x][y]
        for result in results:
            child = container[result[0]][result[1]]
            if bathymetry.is_below_depth((parent.longi, parent.lati),(child.longi, child.lati),DEPTH_LIMIT):
                ind.append(True)
            else:
                ind.append(False)
        ind = np.array(ind)
        results = results[ind]
            
        return tuple(map(tuple, results))
    
    def cost(self, start, end, v, time, container):
        ps = container[start[0]][start[1]]
        pe = container[end[0]][end[1]]
        i_depth = bathymetry.water_depth([ps.longi, ps.lati])
        i_wind_U = weather_info.u([ps.lati, ps.longi, time]) / 0.51444
        i_wind_V = weather_info.v([ps.lati, ps.longi, time]) / 0.51444
        i_Hs = weather_info.hs([ps.lati, ps.longi, time])
        i_Tp = weather_info.tp([ps.lati, ps.longi, time])
        i_head = weather_info.hdg([ps.lati, ps.longi, time])
        i_cu = weather_info.cu([ps.lati, ps.longi, time])
        i_cv = weather_info.cv([ps.lati, ps.longi, time])
        dist, bearing = greatcircle_inverse(ps.longi, ps.lati, pe.longi, pe.lati)
        vfit = ship_info.power_to_speed(v, bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
        timec = dist / vfit / 1.852
        fuel_cost = ship_info.weather2fuel(vfit, bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
        fuelc = fuel_cost[3] * dist / 1.852
        
        return np.array([np.float(fuelc), timec]).ravel(), np.array(fuel_cost + [vfit]).ravel().tolist()

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def dijkstra(graph, container, start, goal, v, q, Initial_time, Initial_fuelc):
    frontier = PriorityQueue()
    frontier.put(start, np.array([Initial_fuelc, Initial_time, 0, 0, 0, 0, v]))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = [Initial_fuelc, Initial_time, 0, 0, 0, 0, v]
    
    while not frontier.empty():
        current = frontier.get()

        if current[0] == goal[0]:
            break
        
        for next in graph.neighbors(current, q, container):
            current_time = cost_so_far[current][1]
            point_cost, power_info = graph.cost(current, next, v, current_time, container)
            new_cost = np.array(cost_so_far[current][:2]) + point_cost
            if next not in cost_so_far or new_cost[0] < cost_so_far[next][0]:
                cost_so_far[next] = new_cost.tolist() + power_info                
                priority = new_cost.tolist()
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

    
def construct_dijpath(goal, start, came_from, cost_so_far, container):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    path_info = []
    for subpath in path:
        child = container[subpath[0]][subpath[1]]
        child_info = cost_so_far[subpath]
        path_info.append([child_info[1], child.longi, child.lati, child_info[0], child_info[2], child_info[3], child_info[4] ,child_info[5], child_info[6]])
    
    return np.array(path_info)

## departure
#p_dep = np.array([3.9, 52.0])
## destination
#p_des = np.array([-5.0, 49.0])
## # construct route 1
#DijkGraph1 = gen_graph(p_dep, p_des, 20, 1, 25, 0.07)
#GraphWidth = len(DijkGraph1)
#GraphHeight = max([len(dijkgraph) for dijkgraph in DijkGraph1])
#DijkGrid1 = DijkstraGrid(GraphWidth, GraphHeight)
#Dcf1, Dcsf1 = dijkstra(DijkGrid1, DijkGraph1, (0, 0), (GraphWidth - 1, 0), 20, 5, 0, 0)
#Dij_path1 = construct_dijpath((GraphWidth - 1, 0),(0, 0), Dcf1, Dcsf1, DijkGraph1)
#dij_timec = Dij_path1[-1,0]
#dij_fuelc = Dij_path1[-1,3]
# departure
p_dep = np.array([-5.0, 49.0])
# destination
p_des = np.array([-73.0, 40.0])
# construct route 2
DijkGraph2 = gen_graph(p_dep, p_des, 20, 6, 25, 0.1)
GraphWidth = len(DijkGraph2)
GraphHeight = max([len(dijkgraph) for dijkgraph in DijkGraph2])
DijkGrid2 = DijkstraGrid(GraphWidth, GraphHeight)
Dcf2, Dcsf2 = dijkstra(DijkGrid2, DijkGraph2, (0, 0), (GraphWidth - 1, 0), 20, 5, 0, 0)
Dij_path2 = construct_dijpath((GraphWidth - 1, 0),(0, 0), Dcf2, Dcsf2, DijkGraph2)

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
#x, y = m(Dij_path1[:,1], Dij_path1[:,2])
#m.plot(x, y, marker=None, linewidth=3, color='g')
#m.scatter(x, y, marker='D',color='g')

x, y = m(Dij_path2[:,1], Dij_path2[:,2])
m.plot(x, y, marker=None, linewidth=3, color='g')
m.scatter(x, y, marker='D',color='g')
m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
m.drawcoastlines()
m.fillcontinents()
plt.show()