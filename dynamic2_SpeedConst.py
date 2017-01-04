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


class dy2_node:
    def __init__(self, longi, lati, stage, num):
        self.longi = longi
        self.lati = lati
        self.stage = np.int(stage)
        self.num = num
        self.spath = [] 
        self.speed = 20.0
        self.time = 0.0
        self.fuelc = 0.0
        self.PE = 0.0
        self.PS =0.0
        self.fuel_kg_per_hour = 0.0
        self.fuel_kg_per_nm = 0.0
        
        

def TwoDDP(departure, destination, v, delta_t, n, eta, q, Initial_time, Initial_fuelc ):
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
    nodeset = {}
    nodeset[1] = np.array([dy2_node(longi_c[0], lati_c[0], 1, (n - 1) / 2)])        
    for i in range(1, K - 1):
        temp_brng = greatcircle_point(longi_c[i - 1], lati_c[i - 1], para_stage[i - 1, 0], para_stage[i - 1, 1])[2]
        temp_point1 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng + 90)
        temp_point2 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng - 90)
        temp_x, temp_y = m.gcpoints(temp_point1[0], temp_point1[1], temp_point2[0], temp_point2[1], n)
        temp_pos =np.array(m(temp_x, temp_y ,inverse = True)).T
        # node number starts at 0
        nodeset[i + 1] = np.array([dy2_node(tempnode[0], tempnode[1], i + 1, nodes) for nodes, tempnode in enumerate(temp_pos)])
    nodeset[K] = np.array([dy2_node(longi_c[-1], lati_c[-1], K, (n - 1) / 2)])

    
    # initialize stage 2
    nodeset[1][0].time = Initial_time
    nodeset[1][0].fuelc = Initial_fuelc
    i_depth = bathymetry.water_depth([nodeset[1][0].longi, nodeset[1][0].lati])
    i_wind_U = weather_info.u([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time]) / 0.51444
    i_wind_V = weather_info.v([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time]) / 0.51444
    i_Hs = weather_info.hs([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time])
    i_Tp = weather_info.tp([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time])
    i_head = weather_info.hdg([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time])
    i_cu = weather_info.cu([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time])
    i_cv = weather_info.cv([nodeset[1][0].lati, nodeset[1][0].longi, nodeset[1][0].time])
    for nodes in nodeset[2]:
        if not (bathymetry.is_below_depth((nodeset[1][0].longi, nodeset[1][0].lati), (nodes.longi, nodes.lati), DEPTH_LIMIT)):
            continue
        else:
            nodes.spath = nodeset[1][0]
            nodes_dist, nodes_bearing = greatcircle_inverse(nodeset[1][0].longi, nodeset[1][0].lati, nodes.longi, nodes.lati)
            nodes_fuelc = ship_info.weather2fuel(v, nodes_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
             # CHECK WHETHER POWER EXCEED [ADDED 2017.1.3]
            if nodes_fuelc[1] > ship_info.Engine:
                vfit = ship_info.speed_fit(v, nodes_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
                nodes_fuelc = ship_info.weather2fuel(vfit, nodes_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
            else:
                vfit = v
            nodes.time = nodes_dist / vfit / 1.852 + nodeset[1][0].time
            nodes.speed = vfit
            nodes.PE = nodes_fuelc[0]
            nodes.PS = nodes_fuelc[1]
            nodes.fuel_kg_per_hour = nodes_fuelc[2]
            nodes.fuel_kg_per_nm = nodes_fuelc[3]
            nodes.fuelc = nodes.fuel_kg_per_nm * nodes_dist / 1.852 + nodeset[1][0].fuelc
    n = len(nodeset[2])
    # stage 3 to final - 1
    for i in range(2, K):
        for nodes in nodeset[i + 1]: 
            if nodes.num - q < 0:
                temp_subset = [subset for subset in nodeset[i][:q + nodes.num + 1]]
            elif (nodes.num - q >= 0) & (nodes.num < n - q):
                temp_subset = [subset for subset in nodeset[i][nodes.num - q:nodes.num + q + 1]]
            else:
                temp_subset = [subset for subset in nodeset[i][nodes.num - q:]]
            temp_subset = np.array(temp_subset)
            mask_cross = np.array([bathymetry.is_below_depth((nodes.longi, nodes.lati),(subset.longi, subset.lati), DEPTH_LIMIT) for subset in temp_subset])
            mask_blank = np.array([subset.spath != [] for subset in temp_subset])
#            mask_cross = ~mask_cross
            temp_subset = temp_subset[np.all((mask_blank, mask_cross), axis = 0)]
            if temp_subset.size == 0:
                continue
            else:
                h = []
                for num1, subset in enumerate(temp_subset):
                    i_depth = bathymetry.water_depth([subset.longi, subset.lati])
                    i_wind_U = weather_info.u([subset.lati, subset.longi, subset.time]) / 0.51444
                    i_wind_V = weather_info.v([subset.lati, subset.longi, subset.time]) / 0.51444
                    i_Hs = weather_info.hs([subset.lati, subset.longi, subset.time])
                    i_Tp = weather_info.tp([subset.lati, subset.longi, subset.time])
                    i_head = weather_info.hdg([subset.lati, subset.longi, subset.time])
                    i_cu = weather_info.cu([subset.lati, subset.longi, subset.time])
                    i_cv = weather_info.cv([subset.lati, subset.longi, subset.time])
                    sub_dist, sub_bearing = greatcircle_inverse(subset.longi, subset.lati, nodes.longi, nodes.lati)
                    #
                    sub_fuelc = ship_info.weather2fuel(v, sub_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
                    if sub_fuelc[1] > ship_info.Engine:
                        vfit = ship_info.speed_fit(v, sub_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
                        sub_fuelc = ship_info.weather2fuel(vfit, sub_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
                    else:
                        vfit = v
                    timec = sub_dist / vfit / 1.852 + subset.time                    
                    fuelc = sub_fuelc[3] * sub_dist / 1.852 + subset.fuelc
                    heapq.heappush(h,[fuelc, vfit, timec, sub_fuelc[0], sub_fuelc[1], sub_fuelc[2], sub_fuelc[3], subset])
                nodes.fuelc, nodes.speed, nodes.time, nodes.PE, nodes.PS, nodes.fuel_kg_per_hour, nodes.fuel_kg_per_nm, nodes.spath = heapq.heappop(h)
                
   
    return nodeset

def construct_dypath2(dy_set):
    K = max(dy_set.keys())
    p = dy_set[K][0]
    path_info = []
    while p.spath != []:
        path_info.append([p.time, p.speed, p.longi, p.lati, p.fuelc, p.PE, p.PS, p.fuel_kg_per_hour, p.fuel_kg_per_nm])
        p = p.spath
    path_info.append([p.time, p.speed, p.longi, p.lati, p.fuelc, p.PE, p.PS, p.fuel_kg_per_hour, p.fuel_kg_per_nm])
    path_info = path_info[::-1]
    
    return np.array(path_info)
    
## departure
#p_dep = np.array([3.9, 52.0])
## destination
#p_des = np.array([-5.0, 49.0])
## # construct route 1
#dy2_set1 = TwoDDP(p_dep, p_des, 20, 1, 31, 0.08, 3, 0, 0)
#dy2_path1 = construct_dypath2(dy2_set1)
#dy2timec = dy2_path1[-1,0]
#dy2fuelc = dy2_path1[-1,3]

dy2timec = 0
dy2fuelc = 0
# departure
p_dep = np.array([-5.0, 49.0])
# destination
p_des = np.array([-73.0, 40.0])
# construct route 2
dy2_set2 = TwoDDP(p_dep, p_des, 20, 6, 31, 0.2, 3, dy2timec, dy2fuelc)
dy2_path2 = construct_dypath2(dy2_set2)

m = Basemap(
  projection="merc",
  resolution='l',
  area_thresh=0.1,
  llcrnrlon=-75,
  llcrnrlat=35,
  urcrnrlon=10,
  urcrnrlat=55
)

#longi = []
#lati = []
#for i in dy2_set1.keys():
#    for j in dy2_set1[i]:
#        longi.append(j.longi)
#        lati.append(j.lati)
#for i in dy2_set2.keys():
#    for j in dy2_set2[i]:
#        longi.append(j.longi)
#        lati.append(j.lati)
#
#longi = np.array(longi)
#lati = np.array(lati)
#plt.figure(figsize=(20,15))
#plt.title('Dynamic Programming Grid', fontsize=20, fontweight='bold')
#x, y = m(longi, lati)
#m.drawcoastlines()
#m.plot(x, y, '.', color='m',markersize = 5)
#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#plt.show()  
plt.figure(figsize=(20, 15))
#x, y = m(dy2_path1[:,1], dy2_path1[:,2])
#m.plot(x, y, marker=None, linewidth=3, color='g')
#m.scatter(x, y, marker='D',color='g')

x, y = m(dy2_path2[:,2], dy2_path2[:,3])
m.plot(x, y, marker=None, linewidth=3, color='g')
m.scatter(x, y, marker='D',color='g')
m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
m.drawcoastlines()
m.fillcontinents()
plt.show()