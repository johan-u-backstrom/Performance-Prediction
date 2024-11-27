'''
The performance_prediction module implements the classes and functions needed to
support multivariable (multi array) Cross Directional (CD) Performance Prediction for the QCS 4.0
web app.

Copyright: Honeywell Process Solutions - North Vancouver
'''

import numpy as np
from source.cd_system import CDSystem
from source.cd_actuator import CDActuator
from source.cd_measurement import CDMeasurement
from source.cd_process_model import CDProcessModel
from source.cd_mpc import CDMPC

class CDPerformancePrediction:
    '''
    The CDPerformancePrediction class holds all methods and data required for 
    performing CD performance predictions based on CD-MPC control. It solves the 
    steady state CD-MPC problem for a given set of CD actuator profiles and 
    CD Measurement profiles and their associated data.

    This is the top level class that the QCS 4.0 RESTful API calls.

    Calling Syntax:
        cd_performance_prediction = CDPerformancePrediction(cd_system_dict, cd_actuators_lst, cd_measurements_lst, 
                                                            cd_process_model_dict, cd_mpc_tuning_dict)

    Input Parameters:
        cd_system  -        A dictionary of (QCS) system data that is not tied to a CD actuator or CD measurement    
        cd_actuators -      A dictionary (if one cd actuator) or a list of dictionaries with CD actuator data
        cd_measurements -   A disctionary (if one cd measurement) or a list of dictionary with CD measurement data
        cd_process_model -  A dictionary of CD process model data
        cd_mpc_tuning -     A dictionary of cd-mpc tuning data
    
    Class Attributes:
        Nu -                    Number of CD actuator beams
        Ny -                    Number of CD measurement arrays
        cd_Actuators -          A List of CDActuator objects
        cd_measurements -       A List of CDMeasurement objects
        cd_process_model -      A CDProcessModel object
        cd_mpc -                A CDMPC object 

    '''

    def __init__(self, cd_system, cd_actuators, cd_measurements, cd_process_model, cd_mpc_tuning):
        '''
        The Class Constructor
        '''
        # Add a system object
        self.cd_system = CDSystem(cd_system)
     
        # Add a List of cd actuator objects
        if type(cd_actuators) == dict:
            # convert to a list of dicts
            cd_actuators = [cd_actuators]
        self.Nu = len(cd_actuators)
        self.cd_actuators = []
        for act_dict in cd_actuators:
            self.cd_actuators.append( CDActuator(act_dict))

        # Add a List of cd measurement objects: cd_measurements
        if type(cd_measurements) == dict:
            cd_measurements = [cd_measurements]
        self.Ny = len(cd_measurements)
        self.cd_measurements = []
        for meas_dict in cd_measurements:
            self.cd_measurements.append(CDMeasurement(meas_dict))

        # Add a cd_process_model object
        self.cd_process_model = CDProcessModel(cd_process_model, self.cd_system, self.cd_actuators, 
                                               self.cd_measurements, self.Nu, self.Ny)
        
        # Add a cd_mpc object
        self.cd_mpc = CDMPC(cd_mpc_tuning, self.cd_system, self.cd_actuators, self.cd_measurements,
                            self.cd_process_model)
    #END constructor

     
                                          

                    


