'''
The performance_prediction module implements the classes and functions needed to
support multivariable (multi array) Cross Directional (CD) Performance Prediction for the QCS 4.0
web app.

Copyright: Honeywell Process Solutions - North Vancouver
'''

import numpy as np
from source.cd_measurement import CDMeasurement
# import source.QP

class CDPerformancePrediction:
    '''
    The CDPerformancePrediction class holds all methods and data required for 
    performing CD performance predictions based on CD-MPC control. It solves the 
    steady state CD-MPC problem for a given set of CD actuator profiles and 
    CD Measurement profiles and their associated data.

    Calling Syntax:
        cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators, cd_measurements)

    Input Parameters:
        syste_dict  -           A dictionary of (QCS) system data that is not tied to a CD actuator or CD measurement    
        cd_actuators_lst -      A List of Dictionaries with CD actuator data
        cd_measurement_lst -    A List of Dictionary with CD measurement data
    
    Class Attributes:
        Nu -                    Number of CD actuator beams
        Ny -                    Number of CD measurement arrays
        cd_measurements -       A List of cdMeasurement objects
    '''

    def __init__(self, system_dict, cd_actuators_lst, cd_measurements_lst):
        '''
        The Class Constructor
        '''
        # Attributes from the system_data Dictionary
        self.sample_time = system_dict.get("sampleTime")
        self.number_of_cd_bins = system_dict.get("numberOfCDBins")
        self.bin_width = system_dict.get("binWidth")
        self.spatial_eng_units = system_dict.get("spatialEngineeringUnits")
        self.spatial_disp_units = system_dict.get("SpatialDisplayUnits")

        # Add a List of cd actuator objects
        # self.cd_actuators = cd_actuators
        self.Nu = len(cd_actuators_lst)

        # Add a List of cd measurement objects: cd_measurements
        self.Ny = len(cd_measurements_lst)
        self.cd_measurements = []
        for meas_dict in cd_measurements_lst:
            self.cd_measurements.append(CDMeasurement(meas_dict))


    #END constructor

     
                                          

                    


