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

    '''

    def __init__(self, cd_measurement_dict):
        '''
        The Class Constructor
        '''
        # Attributes provided by the QCS 4.0 user through the RESTful API
        self.name =  cd_measurement_dict.get('name')
        self.resolution = cd_measurement_dict.get('resolution')
        self.units = cd_measurement_dict.get('units')
        self.control_mode = cd_measurement_dict.get('controlMode')
        self.initial_profile = cd_measurement_dict.get('initialProfile')
        self.initial_profile_spectrum = cd_measurement_dict.get('initialProfileSpectrum')
        self.final_profile = cd_measurement_dict.get('finalProfile')
        self.final_profile_spectrum = cd_measurement_dict.get('finalProfileSpectrum')
        self.low_edge_of_sheet = cd_measurement_dict.get('lowEdgeOfSheet')
        self.high_edge_of_sheet = cd_measurement_dict.get('highEdgeOfSheet')
        self.md_target = cd_measurement_dict.get('mdTarget')
        self.bias_target = cd_measurement_dict.get('biasTarget')
        self.weight = cd_measurement_dict.get('weight')

        # Attributes calculated by class methods
        [self.first_valid_index, self.last_valid_index] = self.calc_first_last_valid_index(self.initial_profile)
        
        #  Attributes set by / called from the CDMPC object
        self.disturbance_profile = []
        self.max_expected_error = None
        self.q1_norm = None
        self.q1 = None
    
        # END Constructor
    
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