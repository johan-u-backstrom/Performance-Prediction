'''
The performance_prediction module implements the classes and functions needed to
support multivariable (multi array) Cross Directional (CD) Performance Prediction for the QCS 4.0
web app.

Copyright: Honeywell Process Solutions - North Vancouver
'''

import numpy as np
import QP

class CDPerformancePrediction:
    '''
    The CDPerformancePrediction class holds all methods and data required for 
    performing CD performance predictions based on CD-MPC control. It solves the 
    steady state CD-MPC problem for a given set of CD actuator profiles and 
    CD Measurement profiles and their associated data.

    Calling Syntax:
        cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators, cd_measurements)

     Input Parameters:
        system_data  -          A dictionary of (QCS) system data that is not tied to a CD actuator or CD measurement    
        cd_actuators -          A dictionary of CD actuator objects
        cd_measurements -       A dictionary of CD measurement objects
    '''

    def __init__(self, system_data, cd_actuators, cd_measurements):
        '''
        The Class Constructor
        '''
        self.sample_time = system_data.get("sample time")
        self.number_of_cd_bins = system_data.get("number of cd bins")
        self.cd_bin_width = system_data.get("cd bin width")
        self.spatial_eng_units = system_data.get("spatial eng units")
        self.spatial_disp_units = system_data.get("spatial disp units")

        self.cd_actuators = cd_actuators
        self.cd_measurements = cd_measurements
    #END constructor
   
     
                                          

                    


