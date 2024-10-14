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

    Class attributes:
    name -                      name of the cd actuator beam
    resolution -                number of cd actuators in the the cd actuator beam
    units -                     units for the stroke of the cd actuator, e.g. z-units (to be verified)
    width_array -               array of actuator zone widths, length = resolution
    low_offset -                cd alignment parameter, see Experion MX CD Control User Manual for details
    high_offset -               cd alignment parameter, see Experion MX CD Control User Manual for details 
    initial_profile -           the initial cd actuator setpoint profile (before steady state CD-MPC prediction)  
    energy_penalty -            user provided penalty for deviating from nominal (desired) actuator setpoints 
    min_setpoint -              minimum allowed setpoint (to be updated to an array of length = resolution)                
    max_setpoint -              maximum allowed setpoint (to be updated to an array of length = resolution)  
    desired_setpoints -         the nominal or desired setpoints
    max_range -                 estimated maximum range (stroke) for a cd actuator in the beam
    q3_norm -                   normalization divisor for the the CD-MPC q3 weight
    q3 -                        the CD-MPC q3 weight that penalizes moves away from the desired setpoints 

    Class Methods:
    calc_q3_norm -              calculates the q3_norm and max_range
    calc_q3 -                   calculates q3
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
        self.max_range = 0.0
        self.q3_norm = 1.0
        self.q3_scaling = 1.0
        self.q3 = 1.0

        print('CDActuator Class Constructor')
        print('CD actuator name:', self.name)
        print('CD actuator resolution =', self.resolution)

        # END Constructor

    def calc_q3_norm(self):
        '''
        Calculates the following attributes:
        q3_norm -           q3 normalization divisor
        max_range -         the related maximum expected range in the 

        Calling syntax:  calc_q3_norm()

        Input parameters:
        None

        Output parameters:
        None

        '''
        max_range_down = np.max(np.abs(self.desired_setpoints - self.min_setpoint))
        max_range_up = np.max(np.abs(self.max_setpoint - self.desired_setpoints))
        max_range = np.max([max_range_down, max_range_up])
        
        self.max_range = max_range
        self.q3_norm = max_range**2

    def calc_q3_scaling(self, R_row_sum_j, Ny):
        '''
        Calculates the q3 scaling divisor that is required to match the static CD Performance Prediction
        with the steady state of the dynamic CD-MPC close loop solution. See ch 6 in James Fan's Ph.D. thesis 
        for details

        Calling syntax:  calc_q3_scaling(R_row_sum, Ny)

        Input parameters:
        R_row_sum_j -       The row sum of the ration matrix R for this cd actuator j, see CDMPC object for
                            details.
        Ny -                Number of measurement arrays

        Output parametersç
        None
        '''
        self.q3_scaling = R_row_sum_j/Ny



    def calc_q3(self):
        '''
        Calculates the weighting q3 for devations from ideal setpoints.
        '''
        # Here the energy penalty is the energy penaly provided by the user
        # and the normalization divisor is calculated in the calc_q3_norm method
        # and the scaling divisor is calculated in calc_q3_scaling
        self.q3 = (self.energy_penalty/self.q3_norm)/self.q3_scaling