'''
The cd_measurement module implements the cdMeasurement class
which holds all the cd measurement data and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np


class CDMeasurement:
    '''
    The CDMeasurement class holds all the cd measurement data and 
    the methods required to process measurement data.

    Calling Syntax:

    Input Parameters:
    cd_measurement_dict - This is a dictionary containing the following cd measurement data

    Class Attributes:
    name -                  name of the measurement
    control_mode -          cd_only, md_only, or cd_and_md, se Experion MX CD Control User Manual for details
    target_profile -        the desired cd profile for the cd performance prediction problem
    error_profile -         the error profile for the cd performance prediction problem

    Class Methods:

    '''

    def __init__(self, cd_measurement_dict):
        '''
        The Class Constructor
        '''
        # Attributes provided by the QCS 4.0 user through the RESTful API
        self.name =  cd_measurement_dict.get('name')
        self.resolution = cd_measurement_dict.get('resolution')
        self.units = cd_measurement_dict.get('units')
        self.control_mode_in = cd_measurement_dict.get('controlMode')
        self.initial_profile = cd_measurement_dict.get('initialProfile')
        self.initial_profile_spectrum = cd_measurement_dict.get('initialProfileSpectrum')
        self.final_profile = cd_measurement_dict.get('finalProfile')
        self.final_profile_spectrum = cd_measurement_dict.get('finalProfileSpectrum')
        self.low_edge_of_sheet = cd_measurement_dict.get('lowEdgeOfSheet')
        self.high_edge_of_sheet = cd_measurement_dict.get('highEdgeOfSheet')
        self.md_target = cd_measurement_dict.get('mdTarget')
        self.bias_target_profile = cd_measurement_dict.get('biasTarget')
        self.weight = cd_measurement_dict.get('weight')

        # Attributes calculated by class methods
        [self.first_valid_index, self.last_valid_index] = self.calc_first_last_valid_index(self.initial_profile)
        self.control_mode = self.update_control_mode(self.control_mode_in)
        self.target_profile = self.calc_target_profile()
        self.error_profile = self.calc_error_profile()

        #  Attributes set by / called from the CDMPC object
        self.disturbance_profile = []
        self.max_expected_error = None
        self.q1_norm = None
        self.q1 = None
    
        # END Constructor
    
    def update_control_mode(self, control_mode_in):
        '''
        updates the control mode from the caller's (0, 1, 2) to (cd_only, md_only, cd_and_md) 

        Calling syntax: control_model = update_control_mode(control_mode_in)

        Inputs:
        control_mode_in -           control mode as an integer from the caller, inherited from the Matlab code

        Outputs:
        control_mode -              control mode as a descriptive string
        '''
        control_mode = -1
        if control_mode_in == 0:
            control_mode = 'cd_only'
        elif control_mode_in == 1:
            control_mode = 'md_only'
        elif control_mode_in == 2:
            control_mode = 'cd_and_md'
        return control_mode
    
    def calc_target_profile(self, y = None):
        '''
        Calculates the target profile. For cd_only, the controller only controls the shape of the 
        cd profile and the average of the cd profile is allowed to float. For md_only, the controller 
        is only concerned with the average of the cd profile. For cd_and_md, the controller is concerned with 
        both the average and the shape of the cd profile. The desired shape is defined by the bias_target_profile
        and the desired cd_Average is defined by the md_target. 
        '''
        ny = self.resolution
        if y == None:
            y = np.zeros(ny)
        target_profile = np.zeros(ny)
        control_mode = self.control_mode
        bias_target_profile = self.bias_target_profile - np.mean(self.bias_target_profile)
        md_target = self.md_target
        
        if control_mode == 'cd_only':
            target_profile = bias_target_profile
        elif control_mode == 'md_only':
            target_profile = md_target*np.ones(ny)
        elif control_mode == 'cd_and_md':
            target_profile = bias_target_profile +  md_target
        return target_profile

    def calc_error_profile(self):
        '''
        Calculates the error profile for the cd performance prediction control problem

        Calling syntax: error_profile = calc_error_profile()

        Inputs:
        None

        Outputs:
        error_profile -         The error profile for the cd performance prediction control problem  
        '''
        N = self.resolution
        error_profile = np.zeros(N)
        target_profile = self.target_profile
        init_meas_profile = self.initial_profile
        control_mode = self.control_mode

        if control_mode == 'cd_only':
            error_profile = (init_meas_profile - np.mean(init_meas_profile)) - target_profile 
        elif control_mode == 'md_only':
            error_profile = np.mean(init_meas_profile)*np.ones(N) - target_profile
        elif control_mode == 'cd_and_md':
            error_profile = init_meas_profile - target_profile

        return error_profile

    
    def set_disturbance_profile(self, y_d):
        # Normally called from the CDMPC object
        self.disturbance_profile = y_d
    
    def set_max_exppected_error(self, max_exp_e):
         # Normally called from the CDMPC object
         self.max_expected_error = max_exp_e

    def calc_max_expected_error(self):
        '''
        Calculates and sets maximum_expected_error, which is simply 2 sigma of 
        the disturbance profile
        '''
        y_d = self.disturbance_profile
        i_start = self.first_valid_index
        i_end = self.last_valid_index
        self.max_expected_error = 2*np.std(y_d[i_start:i_end-1])
    
    def calc_q1_norm(self):
        '''
        Calculates the normalization divisor for the q1 measurement weight
        '''
        self.q1_norm = self.max_expected_error**2

    def calc_q1(self):
        '''
        Calculates the q1 measurement weight
        '''
        self.q1 = self.weight/self.q1_norm
        
    def calc_first_last_valid_index(self, y):
        '''
        Calculates the first and last valid index for a measurement profile.
        CD bins outside the sheet edges can be padded with zeros, NaNs, or 
        the average of the measurement profile. Note that bins are per definition 
        1-indexed, so cd-bin 1 <-> measurement index 0 and cd-bin my <-> measurement index my-1. 
        '''
        my = len(y)
        first_valid_index = 0
        last_valid_index = my-1 
        y_low_edge_pad = y[0]
        y_high_edge_pad = y[my-1]
       
        if y[1] == y[0]:
            # the low edge is padded, find the first valid index
            for i in range(1,my):
                if y[i] != y_low_edge_pad:
                    first_valid_index  = i
                    break
       
        if y[my-2] == y[my-1]:
            # the high edge is padded, find the last valid index
            for i in range(my-2, -1, -1):
                if y[i] != y_high_edge_pad:
                    last_valid_index = i
                    break

        return first_valid_index, last_valid_index