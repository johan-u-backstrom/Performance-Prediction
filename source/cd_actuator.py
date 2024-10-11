'''
The cd_actuator module implements the CDActuator class
which holds all the cd measurement data and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np

class CDActuator:
    '''
    The CDActuator class holds all the cd actuator data and 
    the methods required to process actuator data.

    Calling Syntax:

    Input Parameters:
    cd_actuator_dict - This is a dictionary containing the following cd actuator data

    '''

    def __init__(self, cd_actuator_dict):
        '''
        The Class Constructor
        '''
        self.name =  cd_actuator_dict.get('name')
        self.resolution = cd_actuator_dict.get('resolution')
        self.units = cd_actuator_dict.get('units') 
        self.width_array = cd_actuator_dict.get('widthArray')
        self.low_offset = cd_actuator_dict.get('lowOffset')
        self.high_offset = cd_actuator_dict.get('highOffset')
        self.initial_profile = cd_actuator_dict.get('initialProfile')
        self.energy_penalty = cd_actuator_dict.get('energyPenalty')
        self.min_setpoint = cd_actuator_dict.get('min')
        self.max_setpoint = cd_actuator_dict.get('max')
        self.desired_setpoints = cd_actuator_dict.get('desiredActSetpoints')
        self.q3_norm = 1.0
        self.q3 = 1.0

        print('CDActuator Class Constructor')
        print('CD actuator name:', self.name)
        print('CD actuator resolution =', self.resolution)

        # END Constructor

    def calc_q3_norm(self, R_row_sum, Ny):
        '''
        Calculates the q3 normalization divisor.

        Calling syntax:  calc_q3_norm(R_row_sum, Ny)

        Input parameters:
        R_row_sum -         The row sum of the ration matrix R for this cd actuator, see CDMPC object for
                            details.
        Ny -                Number of measurement arrays
        '''
        max_range_down = np.max(np.abs(self.desired_setpoints - self.min_setpoint))
        max_range_up = np.max(np.abs(self.max_setpoint - self.desired_setpoints))
        max_range = np.max([max_range_down, max_range_up])
        self.q3_norm = max_range**2/(R_row_sum/Ny)

    def calc_q3(self):
        '''
        Calculates the weighting q3 for devations from ideal setpoints.
        '''
        # Here the energy penalty is the energy penaly provided by the user
        # and the normalization divisor is calculated in the calc_q3_norm method
        self.q3 = self.energy_penalty/self.q3_norm