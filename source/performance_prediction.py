'''
The performance_prediction module implements the classes and functions needed to
support multivariable (multi array) Cross Directional (CD) Performance Prediction for the QCS 4.0
web app.

Copyright: Honeywell Process Solutions - North Vancouver
'''

import numpy as np
from cd_measurement import cdMeasurement 
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
        cd_actuators -          A List of Dictionaries with CD actuator data
        cd_measurements -       A List of Dictionary with CD measurement data
    
    Class Attributes:
        Nu -                    Number of CD actuator beams
        Ny -                    Number of CD measurement arrays
        cdMeasurements -        A List of cdMeasurement objects
    '''

    def __init__(self, system_data, cd_actuators, cd_measurements):
        '''
        The Class Constructor
        '''
        # Attributes from the system_data Dictionary
        self.sample_time = system_data.get("sampleTime")
        self.number_of_cd_bins = system_data.get("numberOfCDBins")
        self.bin_width = system_data.get("binWidth")
        self.spatial_eng_units = system_data.get("spatialEngineeringUnits")
        self.spatial_disp_units = system_data.get("SpatialDisplayUnits")

        # Add a List of cd actuator objects
        # self.cd_actuators = cd_actuators
        self.Nu = len(cd_actuators)

        # Add a List of cd measurement objects: cdMeasurements
        # self.cd_measurements = cd_measurements
        self.Ny = len(cd_measurements)
        for m in cd_measurements:
            self.cdMeasurements[m] = cdMeasurement(cd_measurements[m])


    #END constructor

     
                                          

                    


