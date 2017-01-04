import numpy as np
from scipy.integrate import quad, simps
import scipy.special as scs
from scipy.interpolate import interp1d
from scipy import interpolate

"""
This Ship fuel consumption model will include 6 - 7 components: 
0). Shallow water effect and current -- accurate speed over water

1). Calm water resistance:     Rcalm
2). Added resistance in wind:  Rwind
3). Added resistance in wave:  Rwave
4). Propulsive coefficiency to get the break/engine power Pb= Peff/Cp
5). Get parameter for SFOC

6). Consider the uncertainty from the model by ship data analysis

NB: The effect of hull/propeller roughness, set/drifting effect, and Steering
    for course keeping effect, water temperature and Salt content are neglected
    The leeway/drift angle effect was not considered in the added resistance calculation
    In addition, the current is simply considered on their projection to ship course direction


Created by H. Wang, and modified by W, Mao at 2016-10-10
"""
# function R_wind
# if V_ship + V_wind * np.cos(np.radians(wind_heading - ship_heading)): change to 
# ----> if V_ship + V_wind * np.cos(np.radians(wind_heading - ship_heading)) >= 0
# if V_water <= 0: change to 
# ----> V_water = np.where(V_water <= 0, 0.01, V_water)

# function speedGPS2Water
#
class Ship_FCM:
    def __init__(self):
        # STATIC INFORMATION
        self.L = 211.0  # ship length on waterline [m]
        self.Lpp = 202.0  # length between perpendiculars [m]
        self.B = 35.0  # breadth modulded [m]
        self.aftform = 3  # after body form: 1 - V-shaped sections
        #                2 - Normal section shape
        #                3 - U-shaped sections with Hogner stern
        if self.aftform == 1:
            self.C_stern = -10
        elif self.aftform == 2:
            self.C_stern = 0
        elif self.aftform == 3:
            self.C_stern = 10
        # SHIP TYPE [ADDED 2017.1.3]
        self.shiptype = 5
        if self.shiptype == 1:
#            print ('This is a GENERAL CARGO')
            self.Caa = [0.55, 0.90, 1.0, 1.0, 0.9, 0.87, 0.62, 0.45, 0.25, 0.1, -0.1, -0.48, -0.85, -1.0, -1.42, -1.49, -1.38, -0.9, -0.85]
        elif self.shiptype == 2:
#            print ('This is a CONVENTINAL TANKER')
            self.Caa = [0.90, 0.87, 0.8, 0.7, 0.6, 0.5, 0.35, 0.15, 0.05, -0.05, -0.1, -0.21, -0.25, -0.40, -0.55, -0.60, -0.65, -0.70, -0.67 ]
        elif self.shiptype == 3:   # not finished yet
#            print ('This is a Cylindrical bow tanker')
            self.Caa = [0.55, 0.90, 1.0, 1.0, 0.9, 0.87, 0.62, 0.45, 0.25, 0.1, -0.1, -0.48, -0.85, -1.0, -1.42, -1.49, -1.38, -0.9, -0.85]
        elif self.shiptype == 4:   # not finished yet
#            print ('This is a CONTAINER SHIP')
            self.Caa = [0.55, 0.90, 1.0, 1.0, 0.9, 0.87, 0.62, 0.45, 0.25, 0.1, -0.1, -0.48, -0.85, -1.0, -1.42, -1.49, -1.38, -0.9, -0.85]
        elif self.shiptype == 5: 
#            print ('This is a CRUISE FERRY')
            self.Caa = [0.70, 0.72, 0.74, 0.70, 0.25, 0.20, 0.22, 0.0, -0.23, -0.03, 0.05, 0.05, -0.10, -0.25, -0.55, -0.75, -0.80, -0.75, -0.70]
        else : # not finished yet
#            print ('Shiptype == 6: This is a CAR CARRIER')
            self.Caa = [0.55, 0.90, 1.0, 1.0, 0.9, 0.87, 0.62, 0.45, 0.25, 0.1, -0.1, -0.48, -0.85, -1.0, -1.42, -1.49, -1.38, -0.9, -0.85]
        
        self.i_E = 12.08  # half angle of entrance [degrees],if i_E is unknown, put it as 0(zero)
        self.A_T = 16.00  # transom area [m2]
        self.A_BT = 20  # transverse bulb area [m2]
        self.S_APP = 50.00  # wetted area appendages [m2]
        self.K_2 = 1.5  # the appendage resistance factor 1 + k2
        self.h_B = 4.00  # the position of the centre of the transverse area A_BT above the keel line [m]
        self.hasBulbous = True
        self.loadCon = 1
        self.T_A = 6.8  # draught moulded on A.P. [m]
        self.T_F = 6.8  # draught moulded on F.P. [m]

        self.T = 0.5 * (self.T_A + self.T_F)

        self.lcb = -0.75  # longitudinal position of the centre of buoyancy forward of 0.5L as % of L [%]
        # ship parameters
        self.C_M = 0.98  # midship section coefficient
        self.C_WP = 0.75  # waterplane area coefficient
        self.dist = 37500  # displacement volume moulded [m3]

        self.C_B = self.dist / (self.L * self.B * self.T)  # block coefficient
        self.A_M = self.C_M * self.B * self.T
        self.C_P = self.dist / (self.Lpp * self.A_M)  # prismatic coefficient

        self.g = 9.81
        self.v = 1.05e-6  # viscosity at 20C salt water [m2/s]
        self.ro = 1025.0  # density of sea water at 20C [kg/m3]
        self.ro_air = 1.225  # mass density of air [kg/m3]

        # propeller parameters
        self.D = 5.1  # propeller diameter [m]
        self.Z = 4  # blads number
        self.kQ_o = 0.033275  # open water full scal propeller torque coefficients
        self.n = 2.3  # propeller speed [rps]
        self.clearance = 0.30  # clearance of propeller with the keel line [m]
        self.SFOC = 217  # Specific fuel oil consumption [g/kWh]
#        self.Engine = 31200 # Engine power[kw]
        self.Engine = 19800 # Engine power[kw]
        
        # Wet surface area
        self.S = self.L * (2 * self.T + self.B) * np.sqrt(self.C_M) * (0.453 + 0.4425 * self.C_B + (-0.2862) * self.C_M - 0.003467 * self.B / self.T + 0.3696 * self.C_WP) + 2.38 * self.A_BT / self.C_B
        
        # DYNAMIC INFORMATION
        self.V = 10.289  # m/s
        self.fr = self.V  / np.sqrt(self.g * self.L)
        self.re = self.V * self.L / self.v

        para1 = self.B / self.L
        if para1 < 0.11:
            self.c_7 = 0.229577 * pow(para1, 0.33333)
        elif para1 > 0.25:
            self.c_7 = 0.5 - 0.0625 * pow(para1, -1)
        else:
            self.c_7 = para1

        self.c_1 = 2223105 * pow(self.c_7, 3.78613) * pow((self.T / self.B), 1.07961) * pow((90 - self.i_E), -1.37565)
        self.c_3 = 0.56 * pow(self.A_BT, 1.5) / (self.B * self.T * (0.31 * np.sqrt(self.A_BT) + self.T_F - self.h_B))
        self.c_2 = np.exp(-1.89 * np.sqrt(self.c_3))

        if self.T_F > 0.04:
            self.c_4 = 0.04
        else:
            self.c_4 = self.T_F / self.L

        self.c_5 = 1 - 0.8 * self.A_T / (self.B * self.T * self.C_M)

        para2 = self.V / np.sqrt(2 * self.g * self.A_T / (self.B + self.B * self.C_WP))
        if para2 < 5:
            self.c_6 = 0.2 * (1 - 0.2 * para2)
        else:
            self.c_6 = 0

        if self.B / self.T_A < 5:
            self.c_8 = self.B * self.S / (self.L * self.D * self.T_A)
        else:
            self.c_8 = self.S * (7 * self.B / self.T_A - 25) / (self.L * self.D * (self.B / self.T_A - 3))

        if self.c_8 < 28:
            self.c_9 = self.c_8
        else:
            self.c_9 = 32 - 16 / (self.c_8 - 24)

        if self.L / self.B > 5.2:
            self.c_10 = self.B / self.L
        else:
            self.c_10 = 0.25 - 0.003328402 / (self.B / self.L - 0.134615385)

        if self.T_A / self.D < 2:
            self.c_11 = self.T_A / self.D
        else:
            self.c_11 = 0.0833333 * pow(self.T_A / self.D, 3) + 1.33333

        para3 = self.T / self.L
        if para3 > 0.05:
            self.c_12 = pow(para3, 0.2228446)
        elif para3 < 0.002:
            self.c_12 = 0.479948
        else:
            self.c_12 = 48.20 * pow(para3 - 0.02, 2.078) + 0.479948

        self.c_13 = 1 + 0.003 * self.C_stern

        if pow(self.L, 3) / self.dist < 512:
            self.c_15 = -1.69385
        elif pow(self.L, 3) / self.dist > 1727:
            self.c_15 = 0
        else:
            self.c_15 = -1.69385 + (self.L / pow(self.dist, 1.0 / 3) - 8.0) / 2.36

        if self.C_P < 0.8:
            self.c_16 = 8.07981 * self.C_P - 13.8673 * pow(self.C_P, 2) + 6.984388 * pow(self.C_P, 3)
        elif self.C_P > -0.2:
            self.c_16 = 1.73014 - 0.7067 * self.C_P

        if self.L / self.B < 12:
            self.lam = 1.446 * self.C_P - 0.03 * self.L / self.B
        elif self.L / self.B > 12:
            self.lam = 1.446 * self.C_P - 0.36

        self.l_R = self.L * (1 - self.C_P + 0.06 * self.C_P * self.lcb / (4 * self.C_P - 1))
        self.k_1 = self.c_13 * (0.93 + self.c_12 * pow(self.B / self.l_R, 0.92497) * pow(0.95 - self.C_P, -0.521448) * pow(1 - self.C_P + 0.0225 * self.lcb, 0.6906))
        self.m_1 = 0.0140407 * self.L / self.T - 1.75254 * pow(self.dist, 1.0 / 3.0) / self.L - 4.79323 * self.B / self.L - self.c_16
        self.m_2 = self.c_15 * pow(self.C_P, 2) * np.exp(-0.1 * pow(self.fr, -2))
        self.C_A = 0.006 * pow(self.L + 100, -0.16) - 0.00205 + 0.003 * np.sqrt(self.L / 7.5) * pow(self.C_B, 4) * self.c_2 * (0.04 - self.c_4)
        self.C_P1 = 1.45 * self.C_P - 0.315 - 0.0225 * self.lcb

        self.C_F = 0.075 / pow(np.log10(self.re) - 2, 2)
        self.C_V = self.k_1 * self.C_F + self.C_A
        self.w = self.c_9 * self.C_V * self.L / self.T_A * (0.0661875 + 1.21756 * self.c_11 * (self.C_V / (1 - self.C_P1))) + 0.24558 * np.sqrt(self.B / (self.L * (1 - self.C_P1))) - 0.09726 / (0.95 - self.C_P) + 0.11434 / (0.95 - self.C_B) + 0.75 * self.C_stern * self.C_V + 0.002 * self.C_stern
        self.t = 0.001979 * self.L / (self.B - self.B * self.C_P1) + 1.0585 * self.c_10 - 0.00524 - 0.1418 * pow(self.D, 2) / (self.B * self.T) + 0.0015 * self.C_stern


    # 0.1) Speed through water: -- effect due to current 
    def speedGPS2Water(self, V_GPS, heading_ship, current_U, current_V):
        """
        Function speedGPS2Water(V_GPS, heading_ship, current_U, current_V)

            V_GPS          [knots]: Speed over ground (This is the target speed a ship will sail in actual conditions)
            heading_ship  [degree]: Angle between the true North to a ship's course direction
            current_U      [knots]: Horizontal current speed; 
            current_V      [knots]: Vertical current speed.
        
            Output: V_water [knots]: a ship's speed over water

        Convert speed over ground into speed through water for calculation of ship resistance and power
        
        NB: This code has to be run first to update the speed profile.

        """

        V_water = V_GPS - current_U * np.sin(np.deg2rad(heading_ship)) - current_V * np.cos(np.deg2rad(heading_ship)) # Unit: knots
        V_water = np.where(V_water <= 0, 0.01, V_water)

#        if V_water <= 0:
#            print "Warning: The current speed is fast than a ship's required speed!"
#            V_water = 0.01


        # 0,  DYNAMIC INFORMATION
        self.V = V_water * 0.51444  # V unit [m/s]
        self.fr = self.V  / np.sqrt(self.g * self.L)
        self.re = self.V * self.L / self.v

        return V_water

    # 0.2) Speed through water: -- effect due to shallow water effect 
    def speed2shallow(self, V_water, h_waterdepth, draft = 6.8):
        """
        Function: speed2shallow(V_water, h_waterdepth, draft = 6.8):
            V_water     [knots] --  actual ship through water in shallow water operations
            h_waterdepth [m] --  water depth of the sailing region
            draft        [m] --  a ship's draft when sailing

            Output: V_shallow [knots]: Actual ship speed through water to keep the required GPS speed

        Due to the reduction of shallow water on ship speed, a ship has to output more power 
        to keep the same speed as planed:
        V_water_current_shallow --> V_water_current --> V_gps

        Given the same input marine engine power, a ship's speed in shallow water
        will be reduced than that in the open sea.
        The speed reduction in shallow water is computed by the formula given in
        ISO 2015-15016

        """
        spd_water = V_water * 0.5144  # unit [m/s]
        

        imarea_index = self.A_M * draft / 6.8 / np.power(np.abs(h_waterdepth), 2) 
       

        if imarea_index >= 0.05:
            V_shallow = []
            delta_V1 = spd_water * 0.1242 * (imarea_index - 0.05) + 1 - np.power(np.tanh(self.g * np.abs(h_waterdepth) / np.power(spd_water, 2)), 0.5)
            if V_water.size > 1:
                for num, temp_deltav in enumerate(delta_V1):
                    V_test = np.arange(temp_deltav, 2 * temp_deltav, 0.001) + spd_water[num]
                    delta_V = V_test * 0.1242 * (imarea_index - 0.05) + 1 - np.power(np.tanh(9.81 * np.abs(h_waterdepth)/np.power(V_test, 2)), 0.5)
                    V_comp = np.abs(V_test - delta_V - spd_water[num])
                    V_shallow.append((delta_V[np.argmin(V_comp)] + spd_water[num]) /0.5144)
                V_shallow = np.array(V_shallow).ravel()
            else:
                V_test = np.arange(delta_V1, 2 * delta_V1, 0.001) + spd_water
                delta_V = V_test * 0.1242 * (imarea_index - 0.05) + 1 - np.power(np.tanh(9.81 * np.abs(h_waterdepth)/np.power(V_test, 2)), 0.5)
                V_comp = np.abs(V_test - delta_V - spd_water)
                V_shallow = (delta_V[np.argmin(V_comp)] + spd_water) /0.5144
            
           
#            V_test = np.arange(delta_V1, 2 * delta_V1, 0.001) + spd_water
#            delta_V = V_test * 0.1242 * (imarea_index - 0.05) + 1 - np.power(np.tanh(9.81 * h_waterdepth/np.power(V_test, 2)), 0.5)
#            V_comp = np.abs(V_test - delta_V - spd_water)
#            V_shallow = (delta_V[np.argmin(V_comp)] + spd_water) /0.5144
        else:
            delta_V = 0
            V_shallow = spd_water / 0.5144
    
            

        # 0,  DYNAMIC INFORMATION
        self.V = V_shallow * 0.51444  # Unit [m/s]
        self.fr = self.V  / np.sqrt(self.g * self.L)
        self.re = self.V * self.L / self.v

        return V_shallow


    ###############################################################################################################
    # 1) Calm water resistance calculation
    def R_calmwater(self, V_water, draft = 6.8):
        """ 
        Function R_calmwater(self, V_water, draft)

            V_water -- Ship speed through water. Unit: [Knots]
            draft   -- Ship's draft for resistance calculation

            Output: R_calmwater, unit[Ton]

        Get the calm water resistance in open sea conditions
        NB: INPUT speed should be of the unit [knots] 
        """

        # 0,  DYNAMIC INFORMATION
        self.V = V_water * 0.51444  # V unit [m/s]

        self.fr = self.V  / np.sqrt(self.g * self.L)
        self.re = self.V * self.L / self.v

        self.S = self.L * (2 * draft + self.B) * np.sqrt(self.C_M) * (0.453 + 0.4425 * self.C_B + (-0.2862) * self.C_M - 0.003467 * self.B / draft + 0.3696 * self.C_WP) + 2.38 * self.A_BT / self.C_B

        # 1, Ship speed through water: consider the effect of current
        self.C_F = 0.075 / pow(np.log10(self.re) - 2, 2)


        rF = 0.001 * 0.5 * self.ro * np.power(self.V, 2) * self.S * self.C_F  # Unit: [Ton]


        rAPP = 0.001 * 0.5 * self.ro * np.power(self.V, 2) * self.S_APP * self.K_2 * self.C_F

        # 2, Wave resistance [kN]
        d = -0.9  # a constant coefficient
        rW = 0.001 * self.c_1 * self.c_2 * self.c_5 * self.dist * self.ro * self.g * np.exp(self.m_1 * np.power(self.fr, d) + self.m_2 * np.cos(self.lam * np.power(self.fr, -2)))


        # 3, Additional resistance due to the presence of a bubous bow
        if self.hasBulbous:
            p_B = 0.56 * np.sqrt(self.A_BT) / (draft - 1.5 * self.h_B)
            F_ni = self.V * 0.51444 / np.sqrt(self.g * (draft - self.h_B - 0.25 * np.sqrt(self.A_BT)) + 0.15 * np.power(self.V, 2))
            rB = 0.001 * 0.11 * np.exp(-3 * np.power(p_B, -2)) * np.power(F_ni, 3) * np.power(self.A_BT, 1.5) * self.ro * self.g / (1 + np.power(F_ni, 2))
        else:
            rB = 0

        # 4, Additional pressure resistance due to the immersed transom [kN]
        rTR = 0.001 * 0.5 * self.ro * np.power(self.V, 2) * self.A_T * self.c_6
        rA = 0.001 * 0.5 * self.ro * np.power(self.V, 2) * self.S * self.C_A

        # 5, Total calm water resistance of the ship [kN]
        R_calmwater = rF * self.k_1 + rAPP + rW + rB + rTR + rA

        return R_calmwater 


    #############################################################################################################
    # 2), added resistance due to wind
    def R_wind(self, wind_U, wind_V, V_gps, heading_ship):

        """  
        Function R_wind(wind_U, wind_V, V_gps, heading_ship)

            wind_U         [knots]:  wind speed in horizontal direction
            wind_V         [knots]:  wind speed in vertical direction
            V_gps          [knots]:  a ship's actual sailing speed when planning routes. Unit:
            heading_ship  [degree]:  a ship's course heading, 0--> north, 90 --> east, 180 -->south, 270 --> west
            shiptype              :  1, General cargo[Default]
                                     2, Conventional tanker
                                     3, Cylindrical bow tanker
                                     4, Container
                                     5, Cruise ferry
                                     6, Car Carrier
            

            Output: R_wind, Unit: [kN] 

        The wind resistance is calculated based on ISO2015-90156: with empirical formulas 
        based on large amount of wind tunnel test from Fuiwara et al. (1986)
        NB: the side drifting force and angle are not considered in the current investigation

        """

        ship_heading = heading_ship

        ro_air = self.ro_air  # air density in unit [kg/m3]

        V_ship = V_gps * 0.51444
        wind_U = wind_U * 0.5144
        wind_V = wind_V * 0.5144





        # get the apparent wind direction and apparent wind speed (i.e. apparent wind velocity)
        ship_U = V_ship * np.sin(np.radians(ship_heading))
        ship_V = V_ship * np.cos(np.radians(ship_heading))

        appwind_U = wind_U - ship_U
        appwind_V = wind_V - ship_V

        appwind_spd = np.sqrt(np.power(appwind_U, 2) + np.power(appwind_V, 2))
        appwind_hdg = np.degrees(np.arctan2(appwind_U, appwind_V))
        
 

        # relative heading between ship course and wind blowing [degrees]
        heading_ship2wind = np.array(appwind_hdg)
        heading_ship2wind[heading_ship2wind < 0] = heading_ship2wind[heading_ship2wind < 0] + 360



        # relative wind speed [m/s]
        V_relative =  np.array(appwind_spd)       # relative wind speed along ship course [m/s]

#        print 'Apparent wind', heading_ship2wind, V_relative/0.5144


        # ship's transverse projection area including superstrucures [m^2]
        A_transverse = 1500       

        R_wind = 0.001 * 0.5 * ro_air * self.C_AA (heading_ship2wind) * A_transverse * np.power(V_relative, 2) \
                #- 0.001 * 0.5 * ro_air * self.C_AA (0, shiptype) *  A_transverse * np.power(V_ship, 2)

        return R_wind

    # 2.1) get the wind resistance coefficiency
    def C_AA(self, heading_ship2wind):
        """
        Function C_AA(heading_ship2wind)
        
            heading_ship2wind [degree] -- angles between a ship's GPS course heading and wind blowing
            shiptype -- 1, General cargo[Default]
                        2, Conventional tanker
                        3, Cylindrical bow tanker
                        4, Container
                        5, Cruise ferry
                        6, Car Carrier

        Output: C_AA is the wind force coefficiency according to Test from ISO15016 for different ship types

        """
        

        Heading = np.arange(0, 185, 10)




        heading_ship2wind = np.array(heading_ship2wind)
        heading_ship2wind[heading_ship2wind < 0.0001] = 0.0001
        heading_ship2wind[heading_ship2wind > 180] = 360 - heading_ship2wind[heading_ship2wind > 180]

        f_caa = interpolate.interp1d(Heading, self.Caa)
        C_AA = f_caa(heading_ship2wind)

        return C_AA




    # 2.1) get the wind resistance coefficiency
    def C_AA_Fujiwara(self, heading_ship2wind):
        """
        Function C_AA(heading_ship2wind)
        
            heading_ship2wind [degree] -- angles between a ship's GPS course heading and wind blowing

            Output: C_AA is the wind force coefficiency according to Fujihara's formula based on a series of test

        This function will held to compute the wind resistance coefficiency from Fujikara et al. (1986)

        """
        beta10 = 0.922
        beta11 = -0.507
        beta12 = -1.162
        beta20 = -0.018
        beta21 = 5.091
        beta22 = -10.367
        beta23 = 3.011
        beta24 = 0.341

        sigma10 = -0.458
        sigma11 = -3.245
        sigma12 = 2.313
        sigma20 = 1.901
        sigma21 = -12.727
        sigma22 = -24.407
        sigma23 = 40.310
        sigma24 = 5.481

        eta10 = 0.585
        eta11 = 0.906
        eta12 = -3.239
        eta20 = 0.314
        eta21 = 1.117

        # Ship length overall in [m]
        L_OA = 211
        # Ship breadth in [m]
        B = 35
        # Horizontal distance from midship to centre of lateral projected area [m]
        C_MC = -0.5

        # Lateral projected area above the waterline including superstructures [m^2]
        A_LV = 7350
        # Transversal projected area above the waterline including superstructures [m^2]
        A_XV = 1500
        # Lateral projected area of superstrucures above upper deck [m^2]
        A_OD = 300
        # Height of top superstructure (bridge etc.) [m]
        H_BR = 10
        # Height from waterline to centre of lateral projected area A_LV [m]
        H_C = 20
        # Smoothing range in degress, nomrally 10 [degrees]
        mu = 10 
        
        C_LF = np.where(heading_ship2wind < 90,beta10 + beta11 * A_LV / L_OA / B + beta12 * C_MC / L_OA, beta20 + beta21 * B / L_OA + beta22 * H_C / L_OA + beta23 * A_OD / L_OA / L_OA + beta24 * A_XV /np.power(B, 2))
        C_XLI = np.where(heading_ship2wind < 90, sigma10 + sigma11 * A_LV / L_OA / H_BR + sigma12 * A_XV / B / H_BR, sigma20 + sigma21 * A_LV / L_OA / H_BR + sigma22 * A_XV / A_LV + sigma23 * B / L_OA + sigma24 * A_XV /B / H_BR)
        C_ALF = np.where(heading_ship2wind < 90, eta10 + eta11 * A_OD / A_LV + eta12 * B / L_OA, eta20 + eta21 * A_OD / A_LV)

        
        C_AA_Fujiwara = C_LF * np.cos(np.deg2rad(heading_ship2wind)) + C_XLI * (np.sin(np.deg2rad(heading_ship2wind)) -      \
                0.5 * np.sin(np.deg2rad(heading_ship2wind)) * np.power(np.cos(np.deg2rad(heading_ship2wind)), 2)) * \
                np.sin(np.deg2rad(heading_ship2wind)) * np.cos(np.deg2rad(heading_ship2wind)) + C_ALF *             \
                np.sin(np.deg2rad(heading_ship2wind)) * np.power(np.cos(np.deg2rad(heading_ship2wind)), 3)

        return C_AA_Fujiwara

    ##############################################################################################################
    # 3.0) For wave resistance calculation, one needs have correct wave spectrum, here we use Peirson_Moskowtz spectrum
    #       it is also known as ITTC/ISSC wave spectrum for well developed sea
    def PMspec(self, Hs, Tp, w):
        """
        Function PMspec(Hs, Tp, w)

            Hs [m] -- signfiant wave height
            Tp [s] -- peak wave period
            w [rads] -- given circular wave frequency
        

        PMspec return the Pierson-Moskowitz spectral density:
        Generate Pierson-Moskowitz (ITTC) wave spectrum for the given frequency w

        The formula for the PM spectrum is:
        S(w) = 5*Hm0^2/(wp*wn^5)*exp(-5/4*wn^-4), where
        wp = 2*pi/Tp   and    wn = w/wp 

        This is a suitable model for fully developed sea, i.e. a sea state
        where the wind has been blowing long enough over a sufficiently open
        stretch of water, so that the high-frequency waves have reached an
        equilibrium. In the part of the spectrum where the frequency is
        greater than the peak frequency (w>wp), the energy distribution is
        proportional to w^-5.
        The spectrum is identical with ITTC (International Towing Tank 
        Conference), ISSC (International Ship and Offshore Structures Congress) 
        and Bretschneider, wave spectrum given Hm0 and Tm01.
        """ 


        N, M = 5, 4
        B  = (N-1) / M;
        wp = 2 * np.pi / Tp;
        wn = w / wp

        G0 = np.power((N/M), B) * (M / scs.gamma(B)) # Normalizing factor related to Pierson-Moskovitz form
        G1 = np.power(Hs/4, 2)/ wp * G0              # [m^2 s]
        S = G1 * np.power(wn, -N) * np.exp(-(N / M) * np.power(wn, -M))  # here is the wave spectrum for radius frequency w

        return S

    # 3.1) Added resistance due to waves
    def R_wave(self, Hs, Tp, heading_wave, V_water, heading_ship):
        """
        Function R_wave(Hs, Tp, heading_wave, V_water, heading_ship):
                        
            Hs               =   significant wave height of sea conditions [m]
            Tp               =   wave period [s]
            V_water          =   ship's speed through water [knots]
            heading_ship2wave =   heading angle in waves [degree]

            Output: R_wave, Unit [kN]

        It is a modification of STA-1 method since it is only suitable for the head sea operation 
        with limited wave height up to 5 meters, with heading around +- 45 degrees
        The modification is based on 3 steps:
            (1) Take into account the effect of large significant wave height
            (2) Take into account the effect of wave period, speed, heading through encountered wave frequency
            (3) Take into account the wave heading

        NB: Compute the added resistance due to waves based on STA-l method
            It is reasonable for Colormagic since it requires small ship motions

        """ 

        ship_length = self.L
        ros         = self.ro

        # NB: the relative heading between ship/wind and ship/wave is different
        heading_ship2wave =  np.abs(180 - np.abs(heading_ship - heading_wave))
        alpha = heading_ship2wave 


        lbwl        = 100.0      # distance of the bow to 95% of maximum breadth on the waterline
        vs          = V_water * 0.5144  # speed through water [m/s]
        g = self.g
        
        omega = 2 * np.pi / Tp
        k = np.power(omega, 2) / g
        omegaE_0 = omega + k * vs 
        omegaE = omega + k * vs * np.cos(np.radians(alpha))

        # STA-1 method for the added resistance due to waves
        R_wave_STA1 = 1.0 / 16.0 * ros * g * np.power(Hs, 2) * 35 * np.sqrt(35 / lbwl)
        
        # Modification of the original method to take into account the wave heading, period and ship speed.
        # Mod 1: effect due to large significant wave height
        if Hs > 1:
            factor1 =  np.power(Hs* np.sqrt(ship_length / 100), 1.0)   # first guest formula
        else:
            factor1 = np.power(Hs* np.sqrt(ship_length / 100), 0.5)   # first guest formula, need further research

        # Mod 2: effect due to heading & Tp through the encountered wave frequency
        factor2 = np.abs(omegaE) / omegaE_0 * np.power(V_water/20.0, 3.5)

        # Mod 3: effect due to the wave heading
        factor3 = (9.0 +np.cos(np.radians(2*alpha))) / 10.0 /np.power(Hs, 1.15)

        R_wave = R_wave_STA1 * factor1 * factor2 * factor3 * 1e-3


        return R_wave

    # 3.2) Here is an optional wave resistance calculation based on Mauro's empirical formulas
    def R_wave_SR(self, Hs, Tp, V_water, alpha):
        '''
        Function: wave_resistance(Hs, Tp, V_water, alpha)

            Hs = significant wave height of sea conditions [m]
            Tp = wave period [s]
            V_water = ship's speed through water [knots]
            alpha = heading angle in waves [degree]

            Output: R_wave_SR, Unit: [kN]
            
        NB: This is the empirical alternative wave resistance calculation, but it needs a lot of computer efforts to 
            get the final results, and it also rely on the hydrodynamical calculation to get motion RAOs
        '''

        # Initial the basic ship data
        ship_length = self.L
        ros         = self.ro 
        lbwl        = 100.0      # distance of the bow to 95% of maximum breadth on the waterline
        vs          = V_water * 0.5144  # speed through water [m/s]

        # quadratic function parameters for breadth 
        # NB: if x < = 130m, B=35; if x > 130, B(x) = a*x^2 + b*x + c
        a, b, c = -0.001, -0.059, 60
        
        # CHANGE TO RADIAN
        alpha = np.radians(alpha)
        g = 9.81

        # to get the regular waves from the wave spectrum
        dw     = 0.05
        OMEGA  = np.arange(0.35, 2.5, dw)   # NB: when omega is too small, the wave spectrum will be zero
        S_pm   = self.PMspec(Hs, Tp, OMEGA)
        
        # Excluding wave frequency of spectrum = 0

        H_WAVE = np.sqrt(S_pm * 2) 


        RAWM = []
        for omega, h_wave in zip(OMEGA, H_WAVE):

            # Get the wave parameters in one sea state
            k = np.power(omega, 2) / g
            omegaE = omega + k * vs * np.cos(alpha)
            t = omegaE * vs / g    
            k0 = g / np.power(vs, 2)
            ma = k0 * (1 - 2 * t + np.sqrt(1 - 4 * t)) / 2
            mb = k0 * (1 - 2 * t - np.sqrt(1 - 4 * t)) / 2
            mc = - k0 * (1 + 2 * t + np.sqrt(1 + 4 * t)) / 2
            md = - k0 * (1 + 2 * t - np.sqrt(1 + 4 * t)) / 2

            # ESTABLISH QUADRATIC FUNCTION FOR BREADTH --- VARIABLE IS SHIP LENGTH
            breadth = np.poly1d([a, b, c])
            # DERIVATIVE FOR BREADTH FUNTION
            deri_breadth = breadth.deriv()
            # VERTICAL DISPLACEMENT RELATIVE TO WAVES ASSUMING TO BE A CONSTANT
            # NB: This one should be carefully checked to reflect the reality -- run by WASIM software to get the RAOs
            Z = (- np.power(omegaE, 2) + 2.2 * omegaE + 0.1 ) * h_wave


            # SIGMA IS A FUNCTION 
            sigma = - 0.25 / np.pi * (float(Z) *(1j * omegaE * breadth - vs * deri_breadth))
            print sigma
            # CHANGE SIGMA TO QUADRATIC FUNCTIONS' PARAMETERS
            sigma = np.array(sigma)
            print sigma, omegaE, h_wave
            
            # H(m) INNER FUNCTION 
            def period_dist_variable(x, m):
                return (sigma[0] * np.power(x, 2) + sigma[1] * x + sigma[2]) * (np.cos(m * x) + 1j * np.sin(m * x))
            
            # INTEGRATION OF H(m) 
            # NB: for this ship, the breath change only after x exceeding 130m, i.e x belong [130, shiplength]
            def integ_period(m):
                return quad(period_dist_variable, 130.0, ship_length, args = m)[0]
            
            # INTEGRATION FOR WHOLE RESISTANCE FUNCTION        
            def RAWM_variable(m):
                para = np.power(m + k0 * t, 2) * (m * np.cos(alpha) * k) / np.sqrt(np.power(m + k0 * t, 4) - np.power(m * k0, 2)) 
                return np.power(np.abs(integ_period(m)), 2)  * para
            print t, k0

            if t >= 0.25:
                Rawm = 4 * np.pi * ros * (-quad(RAWM_variable, -np.inf, mc)[0] + quad(RAWM_variable, md, np.inf)[0])
            else:
                Rawm = 4 * np.pi * ros * (-quad(RAWM_variable, -np.inf, mc)[0] + quad(RAWM_variable, md, mb)[0] + quad(RAWM_variable, ma, np.inf)[0])
            
            RAWM.append(Rawm)


        RAWM = np.array(RAWM)


        # integration of wave resistance along the whole wave frequency in a sea state
        R_AWM = 2 * np.sum(RAWM * S_pm / np.power(H_WAVE, 2) ) * dw


        R_wave_SR = (R_AWM ) * 1e-9
        return R_wave_SR


    #####################################################################################################################
    # 4) Get the propulsion coefficiency of the ship
    def get_prop_eff(self, rTotal, V_water):
        '''
        Function get_prop_eff(rTotal, V)

            rTotal      [kN]:  a ship's total resistance [kN]  # This has to be confirmed
            V_water  [knots]:  ship speed through water
            

            Output list:
            etaR   = the relative rotative efficiency
            etaO   = the open water efficiency = T_VA/P_D
            etaS   = shafting efficiency = P_D/P_S
            etaH   = the hull efficiency = (1-t) / (1-w)
       
        kQ_o   = the open water propeller torque coefficient
        h      = distance between shaft centre to aft draught [m]
        n      = propeller speed [rps]
        Z      = blads number
        D      = propeller diameter [m]
        
        C_P    = Prismatic Coefficient
        lcb    = longitudinal position of the centre of buoyancy forward of 0.5L as % of L [%]
        
        t      = thrust deduction faction
        w      = effective wake fraction
        '''

        K = 0.2  # constant, for single screw ships, K = 0 to 0.1 for twin screw ships
        T = rTotal * 1e3 / (1 - self.t)
        pressureDifference = 99047  # for sea water of 15 degrees centigrade [N/m2]

        h = self.T_F - self.clearance - 0.5 * self.D

        bladeRatio = K + (1.3 + 0.3 * self.Z) * T / (np.power(self.D, 2) * (pressureDifference + self.ro * self.g * h))
        

        etaR = 0.9922 - 0.05908 * bladeRatio + 0.07424 * (self.C_P - 0.0225 * self.lcb)
        etaS = 0.99
        etaH = (1 - self.t) / (1 - self.w)
        v_A = (1 - self.w) * V_water * 0.5144
        c_075 = 2.073 * bladeRatio * self.D / self.Z
        tPerC_075 = (0.0185 - 0.00125 * self.Z) * self.D / c_075
        kP = 0.00003
        deltaC_D = (2 + 4 * tPerC_075) * (0.003605 - np.power(1.89 + 1.62 * np.log10(c_075 / kP), -2.5))
        kQ_o = self.kQ_o - deltaC_D * 0.25 * c_075 * self.Z / self.D
        q_O = kQ_o * self.ro * np.power(self.n, 2) * np.power(self.D, 5)
        etaO = 0.72

        return etaR, etaO, etaS, etaH


    def weather2fuel(self, V_gps, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft = 6.8, shiptype = 1):
        
        """
        Function weather2fuel(self, V_gps, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft = 6.8)

            V_gps         [knots]:   Ship's sailing GPS (speed over ground) speed for planning
            heading_ship [degree]:   Ship heading relative to GPS sailing
            current_U     [knots]:   Current speed in horizontal direction
            current_V     [knots]:   Current speed in vertical direction
            h_waterdepth      [m]:   Water depth for sailing -- shallow water effect 
            wind_U        [knots]:   Wind speed in horizontal direction
            wind_V        [knots]:   Wind speed in vertical direction
            Hs                [m]:   Significant wave height
            Tp                [s]:   Peak wave period (P-M spectrum)
            heading_wave [degree]:   Wave heading direction 
            draft             [m]:   A ship's sailing draft = 6.8, 

        Outputlist:
            (1), pE           [kWh]:  Effective power for ship operation
            (2), pS           [kWh]:  Required marine engine power (shaft power, which will loss due to propulive, hull, wake ....)
            (3), Fuel_Rate  [kg/nm]:  Required amount of fuel to travel one nautical mile [kg/nm]


        """

        V_water   = self.speedGPS2Water(V_gps, heading_ship, current_U, current_V)
        V_shallow = self.speed2shallow(V_water, h_waterdepth, draft = 6.8)
        V_water   = V_shallow


        R_calm    = self.R_calmwater(V_water, draft = 6.8)
        R_wind    = self.R_wind(wind_U, wind_V, V_gps, heading_ship)
        if (Hs < 0.01) or (Tp < 0.01) :
            R_wave = np.zeros(heading_ship.size)
        else:
            R_wave = self.R_wave(Hs, Tp, heading_wave, V_water, heading_ship)

#        print R_calm
#        print R_wave
#        print R_wind
        

        

        # Total  resistance of the ship [kN]
        rTotal = R_calm + R_wind + R_wave
        

        etaR, etaO, etaS, etaH = self.get_prop_eff(rTotal, V_water)


        # Ship's speed through water for effective power calculation
        V = V_water * 0.51444  # V unit [m/s]



        # Effective power
        pE  = rTotal * V

   
        # The shaft power
        pS_cal = pE / (etaR * etaO * etaS * etaH)
#        if V_gps > 30 * 0.54:
#            pS_res = -2.594e-10 * np.power(V_gps, 10) + 0.015206 * np.power(wind_U, 4.0) + 1.6909e-05 * np.power(wind_V, 6.0) \
#             + 2741.3 * np.power(heading_ship/180*np.pi, 0.29) - 754.8 * np.power(Hs, 2.9) + 26.806 * np.power(Tp, 3.0)
#        else:
#            pS_res = -1375.8 * np.power(V_gps, 0.6) + 1034.7 * np.power(wind_U, 1.0) - 403.21 * np.power(wind_V, 1.0) \
#             + 1375.8 * np.power(heading_ship/180*np.pi, 1.5) - -9917.1 * np.power(Hs, 2.3) + 123.66 * np.power(Tp, 3.0)
        pS = pS_cal* 1.35

        Fuel_kg_per_hour = pS * self.SFOC * 1e-3 

        Fuel_kg_per_nm   = pS * self.SFOC * 1e-3 / V_gps  # kg/nautialMiles in GPS position
        
        return [pE, pS, Fuel_kg_per_hour, Fuel_kg_per_nm]
    
    @staticmethod
    def jonswap(hs, tp, gamma):    
        omega = np.linspace(0.1, np.pi / 2, 50)
        wp = 2 * np.pi / tp
        sigma = np.where(omega < wp, 0.07, 0.09)
        a = np.exp(-0.5 * np.power((omega - wp) / (sigma * omega), 2.0))
        sj = 320 * np.power(hs, 2) * np.power(omega, -5.0) / np.power(tp, 4) * np.exp(-1950 * np.power(omega, -4) / np.power(tp, 4)) * np.power(gamma, a)
        amp = np.sqrt(2 * simps(sj, omega))
        ind = np.argmax(sj)
        wavelength = 2 * np.pi * 9.81 / np.power(omega[ind], 2)
        return sj, amp, wavelength, omega[ind]
    
    def R_WR(self, hs, tp, gamma):
        '''
         From paper[Improved Formula for Estimating Added Resistance of Ships In Engineer Application]
         Hs = significant wave height of sea conditions [m]
         Tp = wave period [s]
         gamma = jonswap spectram parameter
          Output: RAW, Unit: [kN]
        '''
        # kyy is ship pitch k
        kyy = 0.25
        wave_info = self.jonswap(hs, tp, gamma)
        Eta = wave_info[1]
        wavelengh = wave_info[2]
        omega = wave_info[3]
        # wave entry angle
        EntryAngle = np.arctan(self.B / (2 * self.i_E))
        # divided function into different parts
        simp = 1 + 5 * np.sqrt(self.Lpp / wavelengh) * self.fr
        expm = np.power(0.87 / self.C_B, 1 + 4 * np.sqrt(self.fr))

        # RAWR
        Rawr = 2.25 / 2 * self.ro * self.g * self.B * np.power(Eta, 2) * np.power(np.sin(EntryAngle), 2) * simp * expm
        
        
        # long wave parameters
        a1 = 60.3 * pow(self.C_B, 1.34) * pow(0.87 / self.C_B, 1 + self.fr)
        if self.fr < 0.12:
            a2 = 0.0072 + 0.1676 * self.fr
        else:
            a2 = pow(self.fr, 1.5) * np.exp(-3.5 * self.fr)
        # part of mean omega equation
        para1 = np.sqrt(self.Lpp / self.g) * np.power(kyy / self.Lpp, 1.0 / 3) / 1.17

        #print para1
        if self.fr < 0.05:
            mean_omega = para1 * np.power(0.05, 0.143) * omega
        else:
            mean_omega = para1 * np.power(self.fr, 0.143) * omega
        
        if self.C_B < 0.75:
            b1 = np.where(mean_omega <1, 11.0, -8.5)
            d1 = np.where(mean_omega <1, 14.0, -566 * np.power(self.Lpp / self.B, -2.66) * 6)
        else:
            b1 = np.where(mean_omega <1, 11.0, -8.5)
            d1 = np.where(mean_omega <1, 566 * np.power(self.Lpp / self.B, -2.66), -566 * np.power(self.Lpp / self.B, -2.66) * 6)
        # RAWM
        Rawm = 4 * self.ro * self.g * np.power(Eta, 2) * np.power(self.B, 2) / self.Lpp * np.power(mean_omega, b1) * np.exp(b1 / d1 * (1 - np.power(mean_omega, d1))) * a1 * a2
        
        # RAW
        Raw = Rawm + Rawr
        return Raw / 1000

    def speed_reduce(V, BN, disp, headAngleDegree, fr, C_B, loadCon = 1):
        '''
                 reduced_v  = reduced speed [m/s2]
                     V  = ship speed before reduction [m/s]
                    BN  = Beaufort number
                  disp  = displacement volume moulded [m3]
       headAngleDegree  = weather direction [deg]
                    fr  = Froude number
                   C_B  = Block coefficient
               loadCon  = 1 for laden or normal, 0 for ballast
        '''
        if (BN > 12) or (BN < 0):
            return
            # the weather direction reduction factor
        if headAngleDegree < 30:
            cBeta = 0.5 * 3.0
        elif (headAngleDegree >= 30) & (headAngleDegree < 60):
            # cBeta = 0.5 * (2.3 - 0.03 * ((BN - 4)^2));
            # for the euqation above, it comes from the paper of Ruihua LU, 2013,
            # Low Carbon Shipping Conference, London, Voyeage Optimisation:
            # Prediction of ship specific fuel consumption for energy efficient
            # shipping
            cBeta = 0.5 * (1.7 - 0.03 * np.power(BN - 4, 2))
        elif (headAngleDegree >= 60) & (headAngleDegree < 150):
            # cBeta = 0.5 * (1.5 - 0.06 * ((BN - 6)^2));
            # for the euqation above, it comes from the paper of Ruihua LU, 2013,
            # Low Carbon Shipping Conference, London, Voyeage Optimisation:
            # Prediction of ship specific fuel consumption for energy efficient
            # shipping
            cBeta = 0.5 * (0.9 - 0.06 * np.power(BN - 6, 2))
        elif (headAngleDegree >= 150) & (headAngleDegree < 180):
            cBeta = 0.5 * (0.4 - 0.03 * np.power(BN - 8, 2))
    ##    else:
    ##        cBeta = 1
    #    # the factor alpha for block coefficient and Froude number
        alpha = 0 # initialize the alpha
        if not loadCon: # laden or normal condition
            if C_B <= 0.55:
                alpha = 1.7 - 1.4 * fr - 7.4 * np.power(fr,2)
            elif C_B <= 0.60:
                alpha = 2.2 - 2.5 * fr - 9.7 * np.power(fr,2)
            elif C_B <= 0.65:
                alpha = 2.6 - 3.7 * fr - 11.6 * np.power(fr,2)
            elif C_B <= 0.7:
                alpha = 3.1 - 5.3 * fr - 12.4 * np.power(fr,2)
            elif C_B <= 0.75:
                alpha = 2.4 - 10.6 * fr - 9.5 * np.power(fr,2)
            elif C_B <= 0.8:
                alpha = 2.6 - 13.1 * fr - 15.1 * np.power(fr,2)
            elif C_B <= 0.85:
                alpha = 3.1 - 18.7 * fr + 28 * np.power(fr,2)        
        else: # ballast condition
            if C_B <= 0.75:
                alpha = 2.6 - 12.5 * fr - 13.5 * np.power(fr,2)
            elif C_B < 0.8:
                alpha = 3.0 - 16.3 * fr - 21.6 * np.power(fr,2)
            elif C_B < 0.85:
                alpha = 3.4 - 20.9 * fr + 31.8 * np.power(fr,2)   
    #    # ship form coefficient it's different due to ship categories, the value
    #    # used below is suez-max, other types need more information
        if not loadCon: # laden or normal
            # cForm = 0.6*BN + (BN^6.5) / (2.7 * (disp ^ (2 / 3)));
            # for the euqation above, it comes from the paper of Ruihua LU, 2013,
            # Low Carbon Shipping Conference, London, Voyeage Optimisation:
            # Prediction of ship specific fuel consumption for energy efficient
            # shipping
            cForm = 0.5 * BN + np.power(BN, 6.5) / (2.7 * np.power(disp, 2.0 / 3.0))
        else: # ballast condition
            # cForm = 0.8*BN + (BN^6.5) / (2.7 * (disp ^ (2 / 3)));
            # for the euqation above, it comes from the paper of Ruihua LU, 2013,
            # Low Carbon Shipping Conference, London, Voyeage Optimisation:
            # Prediction of ship specific fuel consumption for energy efficient
            # shipping
            cForm = 0.7 * BN + np.power(BN, 6.5) / (2.7 * np.power(disp, 2.0 / 3.0))
        # if it is a containership, for both laden, normal and ballast condition,
        # cForm = 0.7*BN + (BN^6.5) / (22 * (disp ^ (2 / 3)));
        # the percentage of speed loss
        delt_v_percentage = np.abs(0.01 * cBeta * alpha * cForm)
        # the reduced speed
        reduced_v = (1 - delt_v_percentage) * V
        # [m/s]
        return reduced_v
                                               
    
    def speed_fit(self, V_gps, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft = 6.8):
        '''
        find ship speed in certain engine power
        '''
        V = np.linspace(0.3 * V_gps, 1.1 * V_gps, 10)
        PS = np.array(self.weather2fuel(V, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft))[1]


        fitfunc = interp1d(PS, V, kind = 'cubic')
        vfit = fitfunc(self.Engine)
       
        return vfit
    
    def power_to_speed(self, V_gps, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft = 6.8):
        if isinstance(heading_ship, float):
            V = np.linspace(0.3 * V_gps, 1.1 * V_gps, 10)
            PS = np.array(self.weather2fuel(V, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft))[1]
            fitfunc = interp1d(PS, V, kind = 'cubic')
            vfit = fitfunc(self.Engine)
        else:
            vfit = []
            V = np.linspace(0.3 * V_gps, 1.1 * V_gps, 10)
            for heading in heading_ship:
                Ps = np.array(self.weather2fuel(V, heading, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft))[1]
                fitfunc = interp1d(Ps, V, kind = 'cubic')
                vfit.append(fitfunc(self.Engine))
            vfit = np.array(vfit).ravel()
        return vfit
        
        
#    def speed_fit_3D(self, V_gps, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft = 6.8):
#        '''
#        find ship speed in certain engine power
#        '''
#        V = np.linspace(0.3 * V_gps, 1.1 * V_gps, 10)
#        PS = np.array(self.weather2fuel(V, heading_ship, current_U, current_V, h_waterdepth, wind_U, wind_V, Hs, Tp, heading_wave, draft))[1]
#
#
#        fitfunc = interp1d(PS, V, kind = 'cubic')
#        vfit = fitfunc(self.Engine)
#       
#        return vfit
    

