import numpy as np
from GeneralFunction import *
from Ship_FCM import *
import matplotlib.pyplot as plt
#from matplotlib import animation
#
#from datetime import datetime, timedelta


      
        
bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")
# WATER DEPTH LIMITATION (m)
DEPTH_LIMIT = -11
# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
# INITIATE SHIP INFORMATION
ship_info = Ship_FCM()

class isop_node(object):
    def __init__(self, longi, lati, bearing):
        self.longi = longi
        self.lati = lati
        self.bearing = bearing
        self.time = 0.0
        self.PE = 0.0
        self.PS = 0.0
        self.parent = None
        self.fuelc = 0.0
        self.fuel_kg_per_hour = 0.0
        self.fuel_kg_per_nm = 0.0

def isopone(departure, destination, delta_c, m, Vs, fuel_constraint, delta_d, k, Initial_time, Initial_fuelc):
    '''
    delta_c = increment of heading
    m       = number of heading
    Vs      = ship velocity (assuming the engine power is constant)
    delta_t = time interval between two stages
    delta_d = resolution of the isochrone
    k       = number of sectors
    '''
    trimmed = {}
    node_dict = {}
    fuel_limit = fuel_constraint
    # DISTANCE AND BEARING TO THE DESTINATION
    total_dist, init_bearing = greatcircle_inverse(departure[0], departure[1], destination[0], destination[1]) 
    # DEFINE START POINT
    startpoint = isop_node(departure[0], departure[1], init_bearing)
    startpoint.time = Initial_time
    startpoint.fuelc = Initial_fuelc
    # PUT IT INTO THE DICTIONARY
    node_dict[0] = startpoint
    # WEATHER INFORMATION
    i_depth = bathymetry.water_depth([startpoint.longi, startpoint.lati])
    i_wind_U = weather_info.u([startpoint.lati, startpoint.longi, startpoint.time]) / 0.51444
    i_wind_V = weather_info.v([startpoint.lati, startpoint.longi, startpoint.time]) / 0.51444
    i_Hs = weather_info.hs([startpoint.lati, startpoint.longi, startpoint.time])
    i_Tp = weather_info.tp([startpoint.lati, startpoint.longi, startpoint.time])
    i_head = weather_info.hdg([startpoint.lati, startpoint.longi, startpoint.time])
    i_cu = weather_info.cu([startpoint.lati, startpoint.longi, startpoint.time])
    i_cv = weather_info.cv([startpoint.lati, startpoint.longi, startpoint.time])
    # SHIP HEADING
    current_bearing = np.linspace(init_bearing - delta_c * m, init_bearing + delta_c * m , 2 * m + 1)
    # SAME SPEED OVER GPS
    child_speed = Vs
    # OUTPUT: [pE, pS, Fuel_kg_per_hour, Fuel_kg_per_nm]
    child_fuelc = np.array(ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).T
    # TIME CONSUMING FOR EACH HEADING
    child_time = fuel_limit / child_fuelc[:,2]
    # POSITION OF EACH POINT
    child_points = greatcircle_point(startpoint.longi, startpoint.lati, child_speed * child_time * 1.852, current_bearing).T
    # CHECK CROSSLAND OR SHALLOW WATER AREA
    child_landmasks = np.array([bathymetry.is_below_depth((startpoint.longi,startpoint.lati), child_point, DEPTH_LIMIT) for child_point in child_points[:,:-1]]).ravel() 
    # TRIMMING THE POINTS
    trimmed[1] = [[child_point[0], child_point[1], startpoint.longi, startpoint.lati] for child_point in child_points]
    # REGISTER REST POINTS
    node_dict[1] = np.array([isop_node(child_point[0], child_point[1], child_point[2]) for child_point in child_points[child_landmasks]])
    
    for child_num, child_set in enumerate(node_dict[1]):
        child_set.parent = startpoint
        child_set.time = child_time[child_landmasks][child_num] + startpoint.time
        child_set.PE = child_fuelc[child_landmasks][:,0][child_num]
        child_set.PS = child_fuelc[child_landmasks][:,1][child_num]
        child_set.fuel_kg_per_hour = child_fuelc[child_landmasks][:,2][child_num]
        child_set.fuel_kg_per_nm = child_fuelc[child_landmasks][:,3][child_num]
        child_set.fuelc = fuel_limit + startpoint.fuelc
    
    # SECTION NUMBER STARTS AT [1]
    snum = 1
    while(True):
        if node_dict[snum].size == 0:
            print "No point on stage %d." % (snum - 1)
            break
        else:
            # TRIMMED POINTS COLLECTOR
            trimmed[snum + 1] = []
            # SELLECT SECTORS
            delta_s = np.pi / 60 / 180 * delta_d / np.sin(np.pi / 60 / 180 * snum * Vs * np.mean(child_time) / 1.852)
            sub_secs = np.linspace(init_bearing - k * delta_s * 180, init_bearing + k * delta_s * 180, 2 * k + 1)
            # CONTAINER FOR NEXT STAGE
            children_list = []
            parent_set = node_dict[snum] 
            for num_node, child_node in enumerate(parent_set):
                # WEATHER INFORMATION   
                i_depth = bathymetry.water_depth([child_node.longi, child_node.lati])
                i_wind_U = weather_info.u([child_node.lati, child_node.longi, child_node.time]) / 0.51444
                i_wind_V = weather_info.v([child_node.lati, child_node.longi, child_node.time]) / 0.51444
                i_Hs = weather_info.hs([child_node.lati, child_node.longi, child_node.time])
                i_Tp = weather_info.tp([child_node.lati, child_node.longi, child_node.time])
                i_head = weather_info.hdg([child_node.lati, child_node.longi, child_node.time])
                i_cu = weather_info.cu([child_node.lati, child_node.longi, child_node.time])
                i_cv = weather_info.cv([child_node.lati, child_node.longi, child_node.time])
                current_bearing = np.linspace(child_node.bearing - delta_c * m, child_node.bearing + delta_c * m, 2 * m + 1)
                child_speed = Vs
                child_fuelc = np.array(ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).T
                child_timez = fuel_limit / child_fuelc[:,2]
                # child_node: NODES IN PARENTS LIST
                child_points = greatcircle_point(child_node.longi, child_node.lati, child_speed * child_timez * 1.852, current_bearing).T
                # ADD TRIMMED POINT
                trimmed[snum + 1].append([[child_point[0], child_point[1], child_node.longi, child_node.lati] for child_point in child_points])
                # PRIORITY FOR CHOOSING POINTS
                # HEADING OPOSITE TOWARDS DEPARTURE
                child_ind1 = greatcircle_inverse(startpoint.longi, startpoint.lati, child_points[:, 0], child_points[:,1])[1]
                # DISTANCE TO DESTINATION
                child_ind2 = greatcircle_inverse(destination[0], destination[1], child_points[:, 0], child_points[:,1])[0]
                child_index = np.column_stack((child_ind2, child_ind1))
                # [node number, distance, bearing, time, longi, lati, bearing, pE, pS, Fuel_kg_per_hour, Fuel_kg_per_nm]
                child_list = np.column_stack((np.ones(2 * m + 1) * num_node, child_index, child_timez, child_points, child_fuelc))
                children_list.append(child_list)
            children_list = np.array(children_list)
            children_list = children_list.reshape(children_list.size / 11, 11)
            children_list = children_list[(children_list[:,2] >= min(sub_secs)) & (children_list[:,2] <= max(sub_secs))]
                                    
            child_container = []
            for num in range(2 * k):
                child_finds = children_list[(children_list[:,2] >= sub_secs[num]) & (children_list[:,2] < sub_secs[num + 1])]
                if child_finds.size == 0:
                    continue
                else:       
                    child_landmasks = np.array([bathymetry.is_below_depth((parent_set[int(child_point[0])].longi, parent_set[int(child_point[0])].lati), child_point[4:6], DEPTH_LIMIT) for child_point in child_finds]).ravel()
                    child_finds = child_finds[child_landmasks]
                    if child_finds.size == 0:
                        continue
                    else:
                        child_container.append(child_finds[child_finds[:,1].argsort()][0])
            temp = []
            for num, child in enumerate(child_container):
                temp.append(isop_node(child[4],child[5],child[6]))
                num = int(num)
                temp[num].parent = parent_set[int(child_container[num][0])]
                temp[num].time = child_container[num][3] + temp[num].parent.time
                temp[num].PE = child_container[num][7]
                temp[num].PS = child_container[num][8]
                temp[num].fuel_kg_per_hour = child_container[num][9]
                temp[num].fuel_kg_per_nm = child_container[num][10]                
                temp[num].fuelc = fuel_limit + temp[num].parent.fuelc
            temp = np.array(temp)
            node_dict[snum + 1] = temp
            child_container = np.array(child_container)
            rest_dist = greatcircle_inverse(destination[0], destination[1], child_container[:,4], child_container[:,5])[0]
            rest_fuel = min(rest_dist/ Vs / 1.852 * child_container[:,9])
            if rest_fuel < fuel_limit:
                snum = snum + 1
                break
            else:
                snum = snum + 1
    # DEFINE DESTINATION POINT
    dest = isop_node(destination[0], destination[1], 0)
    dest.time = []
    dest.PE = []
    dest.PS = []
    dest.fuel_kg_per_hour =[]
    dest.fuel_kg_per_nm = []
    dest.fuelc = []
    dest.parent = []
    if bathymetry.water_depth(destination) > DEPTH_LIMIT:
         print 'Destination water depth exceeds limitation.'
         return        
    for num, child in enumerate(node_dict[snum]):  
        if not bathymetry.is_below_depth((child.longi, child.lati), (dest.longi, dest.lati) ,DEPTH_LIMIT):
            continue
        current_distance, current_bearing = greatcircle_inverse(child.longi, child.lati, destination[0], destination[1])
        # WEATHER INFORAMATION
        i_depth = bathymetry.water_depth([child.longi, child.lati])
        i_wind_U = weather_info.u([child.lati, child.longi, child.time]) / 0.51444
        i_wind_V = weather_info.v([child.lati, child.longi, child.time]) / 0.51444
        i_Hs = weather_info.hs([child.lati, child.longi, child.time])
        i_Tp = weather_info.tp([child.lati, child.longi, child.time])
        i_head = weather_info.hdg([child.lati, child.longi, child.time])
        i_cu = weather_info.cu([child.lati, child.longi, child.time])
        i_cv = weather_info.cv([child.lati, child.longi, child.time])
        # CONSTANT SPEED
        child_speed = Vs
        # TOTAL TIME CONSUMPTION
        child_time = current_distance / child_speed / 1.852 + child.time
        # FUEL RELATED
        child_fuelc = ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
        dest.time.append(child_time)
        dest.PE.append(child_fuelc[0])
        dest.PS.append(child_fuelc[1])
        dest.fuel_kg_per_hour.append(child_fuelc[2])
        dest.fuel_kg_per_nm.append(child_fuelc[3])
        dest.fuelc.append(child_fuelc[3] * current_distance / 1.852 + child.fuelc)
        dest.parent.append(child)
    node_dict[snum + 1] = dest   
              
    return node_dict

     
tran_fuelc = 0
tran_timec = 0



# departure
p_dep = np.array([-5.0, 49.0])

# destination
p_des = np.array([-73.0, 40.0])
# construct route 2
isop_set = isopone(p_dep, p_des, 1, 20, 20, 20000, 2, 20, tran_timec, tran_fuelc)

def construct_isop_path(isop_set):
    last_set_num = max(isop_set.keys())
    p = isop_set[last_set_num]
    path_info = []    
    for num, nodes in enumerate(p.parent):
        tempset = []
        tempnode = nodes
        while tempnode.parent != None:
            tempset.append([tempnode.time, tempnode.longi, tempnode.lati, tempnode.fuelc, tempnode.PE, tempnode.PS, tempnode.fuel_kg_per_hour, tempnode.fuel_kg_per_nm])
            tempnode = tempnode.parent
        tempset.append([tempnode.time, tempnode.longi, tempnode.lati, tempnode.fuelc, tempnode.PE, tempnode.PS, tempnode.fuel_kg_per_hour, tempnode.fuel_kg_per_nm])
        tempset = tempset[::-1]
        tempset.append([p.time[num], p.longi, p.lati, p.fuelc[num], p.PE[num], p.PS[num], p.fuel_kg_per_hour[num], p.fuel_kg_per_nm[num]])
        path_info.append(tempset)
    return np.array(path_info)
    
isop_path = construct_isop_path(isop_set)
ind2 = int(np.argwhere(isop_path[:,-1,3] == min(isop_path[:,-1,3])).ravel())

m = Basemap(
  projection="merc",
  resolution='l',
  area_thresh=0.1,
  llcrnrlon=-75,
  llcrnrlat=35,
  urcrnrlon=10,
  urcrnrlat=55
)
# drawing
plt.figure(figsize=(21, 15))
for num, ip in enumerate(isop_path):
    longi = ip[:, 1]
    lati = ip[:, 2]
    x, y = m(longi, lati)
    if num == ind2:
        m.plot(x, y, marker=None, linewidth=3, color='g')
        m.scatter(x, y, marker='D',color='g')
    m.plot(x, y, marker=None, color='b')
m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)    
m.drawcoastlines()
m.fillcontinents()
plt.show()

    