import numpy as np
from GeneralFunction import *
from Ship_FCM import *
import matplotlib.pyplot as plt
#from matplotlib import animation
#
#from datetime import datetime, timedelta


      
        
bathymetry = Bathymetry("GEBCO_2014_2D_9.5_54.0_12.5_60.0.nc")
#bathymetry = Bathymetry("GEBCO_2014_2D_-75.0_30.0_10.0_55.0.nc")
# WATER DEPTH LIMITATION (m)
DEPTH_LIMIT = -11
# INITIATE WEATHER CLASS
weather_info = WeatherInfo("Metdata_OsloKiel_2015-11-21-2015-11-27.mat")
#weather_info = WeatherInfo("Metdata_NorthAtlantic_2015-01-01-2015-01-31.mat")
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
    # NUMBER OF SECTIONS
    number_section = np.int(np.floor(total_dist/ (Vs * 1.852 * delta_t)))
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
            if np.mean(rest_dist) < delta_t * Vs * 1.852:
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
    
# departure
p_dep = np.array([10.6425, 59.1640])
# destination
p_des = np.array([11.8563, 56.7513])
# construct route 1
iso_set1, iso_trimmed1 = isochrone(p_dep, p_des, 1, 20, 20, 1, 0.5, 20, 0, 0)
iso_path1 = construct_isopath(iso_set1)  
ind1 = int(np.argwhere(iso_path1[:,-1,3] == min(iso_path1[:,-1,3])).ravel())  

m = Basemap(
  projection="merc",
  resolution='i',
  area_thresh=0.1,
  llcrnrlon=9.5,
  llcrnrlat=54,
  urcrnrlon=12.5,
  urcrnrlat=60
)
plt.figure(figsize=(15, 21))
for num, ip in enumerate(iso_path1):
    longi = ip[:, 1]
    lati = ip[:, 2]
    x, y = m(longi, lati)
    if num == ind1:
        m.plot(x, y, marker=None, linewidth=3, color='g')
        m.scatter(x, y, marker='D',color='g')
    m.plot(x, y, marker=None, color='b') 
m.drawparallels(np.arange(-90.,120.,1.), labels=[1,0,0,0], fontsize=15)
m.drawmeridians(np.arange(-180.,180.,1.), labels=[0,0,0,1], fontsize=15)    
m.drawcoastlines()
m.fillcontinents()
plt.show()

## departure
#p_dep = np.array([3.9, 52.0])
## destination
#p_des = np.array([-5.0, 49.0])
## construct route 1
#iso_set1, iso_trimmed1 = isochrone(p_dep, p_des, 1, 30, 20, 1, 1, 30, 0, 0)
#iso_path1 = construct_isopath(iso_set1)
#ind1 = int(np.argwhere(iso_path1[:,-1,3] == min(iso_path1[:,-1,3])).ravel())
## lowest fuel consumption
#tran_fuelc = iso_path1[ind1][-1, 3]
#tran_timec = iso_path1[ind1][-1, 0]


#tran_fuelc = 0
#tran_timec = 0


#
## departure
#p_dep = np.array([-5.0, 49.0])
## destination
#p_des = np.array([-73.0, 40.0])
## construct route 2
#iso_set2, iso_trimmed2 = isochrone(p_dep, p_des, 1, 20, 20, 6, 2, 20, tran_timec, tran_fuelc)
#iso_path2 = construct_isopath(iso_set2)
#ind2 = int(np.argwhere(iso_path2[:,-1,3] == min(iso_path2[:,-1,3])).ravel())
#
#m = Basemap(
#  projection="merc",
#  resolution='l',
#  area_thresh=0.1,
#  llcrnrlon=-75,
#  llcrnrlat=35,
#  urcrnrlon=10,
#  urcrnrlat=55
#)
# drawing
#plt.figure(figsize=(15, 15))
#for num, ip in enumerate(iso_path1):
#    longi = ip[:, 1]
#    lati = ip[:, 2]
#    x, y = m(longi, lati)
#    if num == ind1:
#        m.plot(x, y, marker=None, linewidth=3, color='g')
#        m.scatter(x, y, marker='D',color='g')
#    m.plot(x, y, marker=None, color='b')
#for num, ip in enumerate(iso_path2):
#    longi = ip[:, 1]
#    lati = ip[:, 2]
#    x, y = m(longi, lati)
#    if num == ind2:
#        m.plot(x, y, marker=None, linewidth=3, color='g')
#        m.scatter(x, y, marker='D',color='g')
##    m.plot(x, y, marker=None, color='b')
#plt.figure(figsize=(15, 15))
#for key, values in iso_trimmed2.iteritems():
#    if key == 1:
#        for value in values:
#            x = [value[0], value[2]]
#            y = [value[1], value[3]]
#            x, y = m(x, y)
#            m.plot(x, y, color = '#cc0000')
#            m.plot(x[1], y[1], marker = 'o', color = 'y')
#    
#    else:
#        for value in values:
#            for valu in value:
#               x = [valu[0], valu[2]]
#               y = [valu[1], valu[3]] 
#               x, y = m(x, y)
#               m.plot(x, y, color = '#cc0000')
#               m.plot(x[1], y[1], marker = 'o', color = 'y')
#
#for num, ip in enumerate(iso_path2):
#    longi = ip[:, 1][:-1]
#    lati = ip[:, 2][:-1]
#    x, y = m(longi, lati)
#    if num == ind2:
#        m.plot(x, y, marker='D', linewidth=3, color='b', markersize = 5)
#    m.plot(x, y, marker='D', linewidth=3, color='b', markersize = 5)
#        m.scatter(x, y, marker='D',color='b')
            


#m.drawparallels(np.arange(-90.,120.,5.), labels=[1,0,0,0], fontsize=15)
#m.drawmeridians(np.arange(-180.,180.,5.), labels=[0,0,0,1], fontsize=15)    
#m.drawcoastlines()
#m.fillcontinents()
#plt.show()
#plt.savefig('isoillu.png')










#
## animation 
#fig = plt.figure(figsize=(20, 15))
#lon, lat = np.meshgrid(weather_info.lon, weather_info.lat)
#x, y = m(lon, lat)
#iso_integr = np.vstack((iso_path1[ind1], iso_path2[ind2]))
#x1, y1 = m(iso_integr[:,1], iso_integr[:,2])
#it = iso_integr[:, 0]
#wt = np.linspace(0, (124 - 1) * 6, 124)
#ani_ind = int(np.argwhere(wt > max(it))[0])
#
#
#txt = plt.title('', fontsize = 20)
#plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
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
#    txt.set_text("Isochrone @ Significant wave height at " + str(pd))
#    cb = plt.colorbar(cs, orientation ='horizontal', fraction=0.1,shrink=.5, pad=0.05)
#    cb.ax.tick_params(labelsize=20) 
#    return [cs]
#
#def animate(i):
#    if i == 0:
#        cs = m.pcolormesh(x, y, weather_info.Hs[:,:,0], shading='flat', cmap=plt.cm.jet)
#        return [cs]
#    cs = m.pcolormesh(x, y, weather_info.Hs[:,:,i], shading='flat', cmap=plt.cm.jet)
#    index = int(max(np.argwhere(it<wt[i])))
#    m.plot(x1[:index + 1],y1[:index + 1] ,marker='o',markersize = 4, linewidth=3, color='#ff0080')
#    md = weather_info.Met['Time'][i][0]
#    pd = datetime.fromordinal(int(md)) + timedelta(days=md%1) - timedelta(days = 366)
#    txt.set_text("Isochrone @ Significant wave height at " + str(pd))
#    return [cs]
#
#anim = animation.FuncAnimation(fig, animate, init_func=init, frames= ani_ind+1, interval=1000, blit=True)
#mywriter = animation.FFMpegWriter(fps=1)
#
#anim.save('isochrone.mp4', writer=mywriter)


#fig = plt.figure(figsize=(15, 6))
#plt.subplot(1,2,1)
#iso_integr = np.vstack((iso_path1[ind1], iso_path2[ind2][1:]))
#plt.bar(iso_integr[:,0],iso_integr[:,6],linewidth=2)
#plt.xlabel("Time [h]", fontsize = 14)
#plt.ylabel("Fuel per hour [kg]", fontsize = 14)
#plt.grid()
#plt.subplot(1,2,2)
#plt.show()