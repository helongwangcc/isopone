import numpy as np
from GeneralFunction import *
from Ship_FCM import *
from numpy.random import rand
#from matplotlib import pyplot as plt

# BATHYMETRIC CLASS
bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")
# WATER DEPTH LIMITATION (m)
DEPTH_LIMIT = -11
# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
# INITIATE SHIP INFORMATION
ship_info = Ship_FCM()

class dy3_node:
    def __init__(self, longi, lati, stage, num):
        self.longi = longi
        self.lati = lati
        self.stage = np.int(stage)
        self.num = num
        self.dtime = []
        self.spath = []   
        self.speed = []
        self.time = 0.0
        self.fuelc = 0.0
        self.PE = 0.0
        self.PS =0.0
        self.fuel_kg_per_hour = 0.0
        self.fuel_kg_per_nm = 0.0

def ThreeDDP(departure, destination, v, delta_t, n, eta, q, Initial_time, Initial_fuelc):
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
    nodeset[1] = np.array([dy3_node(longi_c[0], lati_c[0], 1, (n - 1) / 2)])        
    for i in range(1, K - 1):
        temp_brng = greatcircle_point(longi_c[i - 1], lati_c[i - 1], para_stage[i - 1, 0], para_stage[i - 1, 1])[2]
        temp_point1 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng + 90)
        temp_point2 = greatcircle_point(longi_c[i], lati_c[i], delta_y * (n - 1) / 2, temp_brng - 90)
        temp_x, temp_y = m.gcpoints(temp_point1[0], temp_point1[1], temp_point2[0], temp_point2[1], n)
        temp_pos =np.array(m(temp_x, temp_y ,inverse = True)).T
        # node number starts at 0
        nodeset[i + 1] = np.array([dy3_node(tempnode[0], tempnode[1], i + 1, nodes) for nodes, tempnode in enumerate(temp_pos)])
    nodeset[K] = np.array([dy3_node(longi_c[-1], lati_c[-1], K, (n - 1) / 2)])
    #
    vn = np.linspace(0.6 * v, v, 20)
    nodeset[1][0].time = Initial_time
    nodeset[1][0].dtime = Initial_time
    nodeset[1][0].fuelc = Initial_fuelc
    nodeset[1][0].speed = v
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
            nodes.spath.append(nodeset[1][0])
            nodes_dist, nodes_bearing = greatcircle_inverse(nodeset[1][0].longi, nodeset[1][0].lati, nodes.longi, nodes.lati)
            temp_timec = nodes_dist / vn / 1.852 + nodeset[1][0].dtime
            temp_fuelc = np.array(ship_info.weather2fuel(vn, nodes_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head))
            time_sec = np.linspace(max(temp_timec), min(temp_timec) - min(temp_timec) * rand() / 100, 21)
            temp_ind = []
            for num in range(10, 20):
                ind = np.argwhere((temp_timec <= time_sec[num]) & (temp_timec >= time_sec[num + 1]))
                if ind.size == 0:
                    continue
                else:
                    temp_ind.append(np.argwhere(temp_fuelc[3] == min(temp_fuelc[3][ind])))
            temp_ind = np.array(temp_ind)
            nodes.speed = vn[temp_ind].ravel()
            nodes.dtime = nodeset[1][0].dtime
            nodes.time = temp_timec[temp_ind].ravel()
            nodes.PE = temp_fuelc[0][temp_ind].ravel()
            nodes.PS = temp_fuelc[1][temp_ind].ravel()
            nodes.fuel_kg_per_hour = temp_fuelc[2][temp_ind].ravel()
            nodes.fuel_kg_per_nm = temp_fuelc[3][temp_ind].ravel()
            nodes.fuelc = temp_fuelc[3][temp_ind].ravel() * nodes_dist / 1.852 + Initial_fuelc
            
    for i in range(2, K):
        for countnum, nodes in enumerate(nodeset[i + 1]):
            if nodes.num - q < 0:
                temp_subset = [subset for subset in nodeset[i][:q + nodes.num + 1]]
            elif (nodes.num - q >= 0) & (nodes.num < n - q):
                temp_subset = [subset for subset in nodeset[i][nodes.num - q:nodes.num + q + 1]]
            else:
                temp_subset = [subset for subset in nodeset[i][nodes.num - q:]]
            temp_subset = np.array(temp_subset)
            mask_cross = np.array([bathymetry.is_below_depth((nodes.longi, nodes.lati),(subset.longi, subset.lati), DEPTH_LIMIT) for subset in temp_subset])
            mask_blank = np.array([subset.spath != [] for subset in temp_subset])
            temp_subset = temp_subset[np.all((mask_blank, mask_cross), axis=0)]
            if temp_subset.size == 0:
                continue
            else:
                timec = []
                timed = []
                fuelc = []
                speedc = []
                pathc = []
                PE = []
                PS = []
                fuel_kg_per_hour = []
                fuel_kg_per_nm = []
                for subset in temp_subset:
                    for num in range(len(subset.time)):
                        i_depth = bathymetry.water_depth([subset.longi, subset.lati])
                        i_wind_U = weather_info.u([subset.lati, subset.longi, subset.time[num]]) / 0.51444
                        i_wind_V = weather_info.v([subset.lati, subset.longi, subset.time[num]]) / 0.51444
                        i_Hs = weather_info.hs([subset.lati, subset.longi, subset.time[num]])
                        i_Tp = weather_info.tp([subset.lati, subset.longi, subset.time[num]])
                        i_head = weather_info.hdg([subset.lati, subset.longi, subset.time[num]])
                        i_cu = weather_info.cu([subset.lati, subset.longi, subset.time[num]])
                        i_cv = weather_info.cv([subset.lati, subset.longi, subset.time[num]])
                        sub_dist, sub_bearing = greatcircle_inverse(subset.longi, subset.lati, nodes.longi, nodes.lati)
                        sub_fuelc = np.array(ship_info.weather2fuel(vn, sub_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head))
                        pathc.append([[subset] for setnum in range(len(vn))])
                        speedc.append(vn)
                        timec.append(sub_dist / vn / 1.852 + subset.time[num])
                        timed.append([subset.time[num] for setnum in range(len(vn))])
                        PE.append(sub_fuelc[0])
                        PS.append(sub_fuelc[1])
                        fuel_kg_per_hour.append(sub_fuelc[2])
                        fuel_kg_per_nm.append(sub_fuelc[3])
                        fuelc.append(sub_fuelc[3] * sub_dist / 1.852 + subset.fuelc[num])
                pathc = np.array(pathc).ravel()
                timec = np.array(timec).ravel()
                timed = np.array(timed).ravel()
                speedc = np.array(speedc).ravel()
                PE = np.array(PE).ravel()
                PS = np.array(PS).ravel()
                fuel_kg_per_hour = np.array(fuel_kg_per_hour).ravel()
                fuel_kg_per_nm = np.array(fuel_kg_per_nm).ravel()
                fuelc = np.array(fuelc).ravel()
                time_sec = np.linspace(max(timec), min(timec) - min(timec) * rand() / 100, 21)
                temp_ind = []
                for num in range(10,20):
                    ind = np.argwhere((timec <= time_sec[num]) & (timec >= time_sec[num + 1]))
                    if ind.size == 0:
                        continue
                    else:
                        temp_ind.append(np.argwhere(fuelc == min(fuelc[ind])))
                temp_ind = np.array(temp_ind)
                nodes.spath = pathc[temp_ind].ravel().tolist()
                nodes.speed = speedc[temp_ind].ravel()
                nodes.dtime = timed[temp_ind].ravel()
                nodes.time = timec[temp_ind].ravel()
                nodes.PE = PE[temp_ind].ravel()
                nodes.PS = PS[temp_ind].ravel()
                nodes.fuel_kg_per_hour = fuel_kg_per_hour[temp_ind].ravel()
                nodes.fuel_kg_per_nm = fuel_kg_per_nm[temp_ind].ravel()
                nodes.fuelc = fuelc[temp_ind].ravel()      

    return nodeset
     

def construct_dypath3(dy_set3):
    K = max(dy_set3.keys())
    p = dy_set3[K][0]     
    path_info = []
    while K - 2 > 0:
        ind = np.argwhere(p.fuelc == min(p.fuelc)).ravel()
        ind = int(ind)
        minfuel = np.float(p.fuelc[ind])
        mintime = np.float(p.time[ind])
        minspeed = np.float(p.speed[ind])
        minpe = np.float(p.PE[ind])
        minps = np.float(p.PS[ind])
        minfhour = np.float(p.fuel_kg_per_hour[ind])
        minfnm = np.float(p.fuel_kg_per_nm[ind])
        path_info.append([mintime, p.longi, p.lati, minfuel, minspeed, minpe, minps, minfhour, minfnm])
        p = p.spath[ind]
        K = K - 1
    ind = np.argwhere(p.fuelc == min(p.fuelc)).ravel()
    ind = int(ind)
    minfuel = np.float(p.fuelc[ind])
    mintime = np.float(p.time[ind])
    minspeed = np.float(p.speed[ind])
    minpe = np.float(p.PE[ind])
    minps = np.float(p.PS[ind])
    minfhour = np.float(p.fuel_kg_per_hour[ind])
    minfnm = np.float(p.fuel_kg_per_nm[ind])
    path_info.append([mintime, p.longi, p.lati, minfuel, minspeed, minpe, minps, minfhour, minfnm])
    p = p.spath[0]
    path_info.append([ p.time, p.longi, p.lati, p.fuelc, p.speed, p.PE, p.PS, p.fuel_kg_per_hour, p.fuel_kg_per_nm])
    path_info = path_info[::-1]
    
    return np.array(path_info)
    
## departure
#p_dep = np.array([3.9, 52.0])
## destination
#p_des = np.array([-5.0, 49.0])
## # construct route 1
#dy3_set1 = ThreeDDP(p_dep, p_des, 20, 1, 31, 0.08, 3, 0, 0)
#dy3_path1 = construct_dypath3(dy3_set1)
#dy3timec = dy3_path1[-1,0]
#dy3fuelc = dy3_path1[-1,3]

dy3timec = 0
dy3fuelc = 0

# departure
p_dep = np.array([-5.0, 49.0])
# destination
p_des = np.array([-73.0, 40.0])
# construct route 2
dy3_set2 = ThreeDDP(p_dep, p_des, 20, 6, 31, 0.1, 3, dy3timec, dy3fuelc)
dy3_path2 = construct_dypath3(dy3_set2)

#plt.figure(figsize=(20, 15))
#x, y = m(dy3_path1[:,1], dy3_path1[:,2])
#m.plot(x, y, marker=None, linewidth=3, color='g')
#m.scatter(x, y, marker='D',color='g')
#
#x, y = m(dy3_path2[:,1], dy3_path2[:,2])
#m.plot(x, y, marker=None, linewidth=3, color='g')
#m.scatter(x, y, marker='D',color='g')
#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)
#m.drawcoastlines()
#m.fillcontinents()
#plt.show()