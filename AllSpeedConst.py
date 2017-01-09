import numpy as np
from GeneralFunction import *
from Ship_FCM import *
import heapq
from numpy.random import rand

# BATHYMETRIC CLASS
bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")
# WATER DEPTH LIMITATION (m)
DEPTH_LIMIT = -11
# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
# INITIATE SHIP INFORMATION
ship_info = Ship_FCM()

class iso_node(object):
    def __init__(self, longi, lati, bearing):
        self.longi = longi
        self.lati = lati
        self.bearing = bearing
        self.speed = 20
        self.time = 0.0
        self.PE = 0.0
        self.PS = 0.0
        self.parent = None
        self.fuelc = 0.0
        self.fuel_kg_per_hour = 0.0
        self.fuel_kg_per_nm = 0.0


def isochrone(departure, destination, delta_c, m, Vs, delta_t, delta_d, k, Initial_time, Initial_fuelc):
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
    # DISTANCE AND BEARING TO THE DESTINATION
    total_dist, init_bearing = greatcircle_inverse(departure[0], departure[1], destination[0], destination[1]) 
    # DEFINE START POINT
    startpoint = iso_node(departure[0], departure[1], init_bearing)
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
    # CHECK WHETHER POWER EXCEED [ADDED 2017.1.3]
    check_index = np.argwhere(child_fuelc[:,1] > ship_info.Engine).ravel()
    #  RECALCULATE FUEL CONSUMPTION
    if check_index.size == 0:
        pass
    else:
        child_speed = np.ones(current_bearing.size) * Vs
        for indc in check_index:
            indc = int(indc)
            child_speed[indc] = ship_info.speed_fit(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
            child_fuelc[indc] = np.array(ship_info.weather2fuel(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).ravel()       
    # BRANCH POINTS WITH DIFFERENT HEADING ANGLE
    child_points = greatcircle_point(startpoint.longi, startpoint.lati, child_speed * delta_t * 1.852, current_bearing).T
#    child_distance = greatcircle_inverse(startpoint.longi, startpoint.lati,child_points[:,0], child_points[:,1])    
    # DEFINE 2ND STAGE  
    child_landmasks = np.array([bathymetry.is_below_depth((startpoint.longi,startpoint.lati), child_point, DEPTH_LIMIT) for child_point in child_points[:,:-1]]).ravel()    
    # ADD TRIMMED POINTS
    trimmed[1] = [[child_point[0], child_point[1], startpoint.longi, startpoint.lati] for child_point in child_points]
    node_dict[1] = np.array([iso_node(child_point[0], child_point[1], child_point[2]) for child_point in child_points[child_landmasks]])
    for child_num, child_set in enumerate(node_dict[1]):
        child_set.parent = startpoint
        child_set.time = delta_t + startpoint.time
        if check_index.size == 0:
            child_set.speed = child_speed
        else:
            child_set.speed = child_speed[child_landmasks][child_num]
        child_set.PE = child_fuelc[child_landmasks][:,0][child_num]
        child_set.PS = child_fuelc[child_landmasks][:,1][child_num]
        child_set.fuel_kg_per_hour = child_fuelc[child_landmasks][:,2][child_num]
        child_set.fuel_kg_per_nm = child_fuelc[child_landmasks][:,3][child_num]
        child_set.fuelc = child_set.fuel_kg_per_hour * delta_t + startpoint.fuelc
        
        
    snum = 2
    while(True):        
#    for snum in range(2, number_section):
        # IF NO POINT IN STAGE ONE THEN BREAK
        if node_dict[snum - 1].size == 0:
            print "No point on stage %d." % (snum - 1)
            break
        else:
            trimmed[snum] = []
            # SECTION FOR TRIMMING UNNECESSARY WAY POINTS
            delta_s = np.pi / 60 / 180 * delta_d / np.sin(np.pi / 60 / 180 * snum * delta_t * Vs / 1.854) # in radian
            sub_secs = np.linspace(init_bearing - k * delta_s * 180, init_bearing + k * delta_s * 180, 2 * k + 1)
            # CONTAINER FOR NEXT STAGE
            children_list = []
            parent_set = node_dict[snum - 1] 
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
                # FUEL CONSUMPTION [CHANGED 2017.1.3]
                child_fuelc = np.array(ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).T
                # CHECK WHETHER POWER EXCEED [ADDED 2017.1.3]
                check_index = np.argwhere(child_fuelc[:,1] > ship_info.Engine).ravel()
                # RECALCULATE FUEL CONSUMPTION
                if check_index.size == 0:
                    pass
                else:
                    child_speed = np.ones(current_bearing.size) * Vs
                    for indc in check_index:
                        indc = int(indc)
                        child_speed[indc] = ship_info.speed_fit(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
                        child_fuelc[indc] = np.array(ship_info.weather2fuel(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).ravel()
                # TRAVELLED DISTANCE                    
                child_points = greatcircle_point(child_node.longi, child_node.lati, child_speed * delta_t * 1.852, current_bearing).T
                # ADD TRIMMED POINT
                trimmed[snum].append([[child_point[0], child_point[1], child_node.longi, child_node.lati] for child_point in child_points])
                # fuelconsumption as criteron
                child_index = greatcircle_inverse(startpoint.longi, startpoint.lati, child_points[:, 0], child_points[:,1]).T                
                # [node number, distance, bearing, longi, lati, bearing, pE, pS, Fuel_kg_per_hour, Fuel_kg_per_nm, speed]
                if check_index.size == 0:
                    child_list = np.column_stack((np.ones(2 * m + 1) * num_node, child_index, child_points, child_fuelc, child_speed * np.ones(current_bearing.size)))
                else:
                    child_list = np.column_stack((np.ones(2 * m + 1) * num_node, child_index, child_points, child_fuelc, child_speed))
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
                    child_landmasks = np.array([bathymetry.is_below_depth((parent_set[int(child_point[0])].longi, parent_set[int(child_point[0])].lati), child_point[3:5], DEPTH_LIMIT) for child_point in child_finds]).ravel()
                    child_finds = child_finds[child_landmasks]
                    if child_finds.size == 0:
                        continue
                    else:
                        child_container.append(child_finds[child_finds[:,8].argsort()][0])
#                        child_container.append(child_finds[child_finds[:,1].argsort()[::-1]][0])
            temp = []
            for num, child in enumerate(child_container):
                temp.append(iso_node(child[3],child[4],child[5]))
                num = int(num)
                temp[num].parent = parent_set[int(child_container[num][0])]
                temp[num].time = delta_t + temp[num].parent.time
                temp[num].speed = child_container[num][10]
                temp[num].PE = child_container[num][6]
                temp[num].PS = child_container[num][7]
                temp[num].fuel_kg_per_hour = child_container[num][8]
                temp[num].fuel_kg_per_nm = child_container[num][9]                
                temp[num].fuelc = child_container[num][8] * delta_t + temp[num].parent.fuelc
            temp = np.array(temp)
            node_dict[snum] = temp
            child_container = np.array(child_container)
            rest_dist = greatcircle_inverse(destination[0], destination[1], child_container[:,3], child_container[:,4])[0]
            if np.min(rest_dist) < delta_t * Vs * 1.852:
                break
            else:
                snum = snum + 1
            
            
    # DEFINE DESTINATION POINT
    dest = iso_node(destination[0], destination[1], 0)
    dest.time = []
    dest.speed = []
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
        # OUTPUT: [pE, pS, Fuel_kg_per_hour, Fuel_kg_per_nm]
        child_fuelc = np.array(ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head))
        # RECALCULATE FUEL CONSUMPTION
        if child_fuelc[1] <= ship_info.Engine:
            pass
        else:
            child_speed = ship_info.speed_fit(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
            child_fuelc = np.array(ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).ravel()       
        # TOTAL TIME CONSUMPTION
        child_time = current_distance / child_speed / 1.852 + child.time
        dest.time.append(child_time)
        dest.speed.append(child_speed)
        dest.PE.append(child_fuelc[0])
        dest.PS.append(child_fuelc[1])
        dest.fuel_kg_per_hour.append(child_fuelc[2])
        dest.fuel_kg_per_nm.append(child_fuelc[3])
        dest.fuelc.append(child_fuelc[3] * current_distance / 1.852 + child.fuelc)
        dest.parent.append(child)
    node_dict[snum + 1] = dest   
               
    return node_dict, trimmed      


def construct_isopath(iso_set):
    last_set_num = max(iso_set.keys())
    p = iso_set[last_set_num]
    path_info = []    
    for num, nodes in enumerate(p.parent):
        tempset = []
        tempnode = nodes
        while tempnode.parent != None:
            tempset.append([tempnode.time, tempnode.longi, tempnode.lati, tempnode.fuelc, tempnode.PE, tempnode.PS, tempnode.fuel_kg_per_hour, tempnode.fuel_kg_per_nm, tempnode.speed])
            tempnode = tempnode.parent
        tempset.append([tempnode.time, tempnode.longi, tempnode.lati, tempnode.fuelc, tempnode.PE, tempnode.PS, tempnode.fuel_kg_per_hour, tempnode.fuel_kg_per_nm, tempnode.speed])
        tempset = tempset[::-1]
        tempset.append([p.time[num], p.longi, p.lati, p.fuelc[num], p.PE[num], p.PS[num], p.fuel_kg_per_hour[num], p.fuel_kg_per_nm[num], p.speed[num]])
        path_info.append(tempset)
    return np.array(path_info)
    
class isop_node(object):
    def __init__(self, longi, lati, bearing):
        self.longi = longi
        self.lati = lati
        self.bearing = bearing
        self.speed = 20.0
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
    # CHECK WHETHER POWER EXCEED [ADDED 2017.1.3]
    check_index = np.argwhere(child_fuelc[:,1] > ship_info.Engine).ravel() 
    #  RECALCULATE FUEL CONSUMPTION
    if check_index.size == 0:
        pass
    else:
        child_speed = np.ones(current_bearing.size) * Vs
        for indc in check_index:
            indc = int(indc)
            child_speed[indc] = ship_info.speed_fit(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
            child_fuelc[indc] = np.array(ship_info.weather2fuel(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).ravel() 
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
        child_set.speed = child_speed[child_landmasks][child_num]
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
                # CHECK WHETHER POWER EXCEED [ADDED 2017.1.3]
                check_index = np.argwhere(child_fuelc[:,1] > ship_info.Engine).ravel()
                # RECALCULATE FUEL CONSUMPTION
                if check_index.size == 0:
                    pass
                else:
                    child_speed = np.ones(current_bearing.size) * Vs
                    for indc in check_index:
                        indc = int(indc)
                        child_speed[indc] = ship_info.speed_fit(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
                        child_fuelc[indc] = np.array(ship_info.weather2fuel(child_speed[indc], current_bearing[indc], i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).ravel()
                # TRAVELLED DISTANCE                    
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
                if check_index.size == 0:
                    child_list = np.column_stack((np.ones(2 * m + 1) * num_node, child_index, child_timez, child_points, child_fuelc, child_speed * np.ones(current_bearing.size)))
                else:
                    child_list = np.column_stack((np.ones(2 * m + 1) * num_node, child_index, child_timez, child_points, child_fuelc, child_speed))
                children_list.append(child_list)
            children_list = np.array(children_list)
            children_list = children_list.reshape(children_list.size / 12, 12)
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
                temp[num].speed = child_container[num][11]
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
    dest.speed = []
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
        # FUEL RELATED
        child_fuelc = ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
        if child_fuelc[1] <= ship_info.Engine:
            pass
        else:
            child_speed = ship_info.speed_fit(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
            child_fuelc = np.array(ship_info.weather2fuel(child_speed, current_bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)).ravel()
        # TOTAL TIME CONSUMPTION
        child_time = current_distance / child_speed / 1.852 + child.time

        dest.time.append(child_time)
        dest.speed.append(child_speed)
        dest.PE.append(child_fuelc[0])
        dest.PS.append(child_fuelc[1])
        dest.fuel_kg_per_hour.append(child_fuelc[2])
        dest.fuel_kg_per_nm.append(child_fuelc[3])
        dest.fuelc.append(child_fuelc[3] * current_distance / 1.852 + child.fuelc)
        dest.parent.append(child)
    node_dict[snum + 1] = dest   
              
    return node_dict, trimmed

def construct_isop_path(isop_set):
    last_set_num = max(isop_set.keys())
    p = isop_set[last_set_num]
    path_info = []    
    for num, nodes in enumerate(p.parent):
        tempset = []
        tempnode = nodes
        while tempnode.parent != None:
            tempset.append([tempnode.time, tempnode.longi, tempnode.lati, tempnode.fuelc, tempnode.PE, tempnode.PS, tempnode.fuel_kg_per_hour, tempnode.fuel_kg_per_nm, tempnode.speed])
            tempnode = tempnode.parent
        tempset.append([tempnode.time, tempnode.longi, tempnode.lati, tempnode.fuelc, tempnode.PE, tempnode.PS, tempnode.fuel_kg_per_hour, tempnode.fuel_kg_per_nm, tempnode.speed])
        tempset = tempset[::-1]
        tempset.append([p.time[num], p.longi, p.lati, p.fuelc[num], p.PE[num], p.PS[num], p.fuel_kg_per_hour[num], p.fuel_kg_per_nm[num], p.speed[num]])
        path_info.append(tempset)
    return np.array(path_info)

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
        fuel_cost = ship_info.weather2fuel(v, bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)        
        if fuel_cost[1] > ship_info.Engine:
            vfit = ship_info.speed_fit(v, bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)
            fuel_cost = ship_info.weather2fuel(vfit, bearing, i_cu, i_cv, i_depth, i_wind_U, i_wind_V, i_Hs, i_Tp, i_head)        
        else:
            vfit = v            
        timec = dist / vfit / 1.852
        
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
            # OUTPUT:[FUEL, TIME], TYPE: NP.ARRAY; [PE, PS, FUEL_TIME, FUEL_DIST, SPEED], TYPE: LIST
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