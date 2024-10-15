'''
The cd_actuator module implements the CDActuator class
which holds all the cd measurement data and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np
from scipy.linalg import toeplitz 

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
    calc_max_range              calculates maximum expected range
    calc_q_scaling              calculates the scaling of the q3 and q4 weights to match static prediction with steady state CD-MPC
    update_q_scaling            updates the q_scaling attribute
    calc_q3_norm                calculates the q3_norm
    calc_q3                     calculates q3
    calc_bending_matrix         calculates the bending matrix (2nd order difference)
    calc_q4_norm                calculates the q4_norm

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
        self.picketing_penalty = cd_actuator_dict.get('picketingPenalty')
        self.min_setpoint = cd_actuator_dict.get('min')
        self.max_setpoint = cd_actuator_dict.get('max')
        self.desired_setpoints = cd_actuator_dict.get('desiredActSetpoints')
        self.bend_limit_first_order = cd_actuator_dict.get('bendLimitFirstOrder')
        self.bend_limit_second_order = cd_actuator_dict.get('bendLimitSecondOrder')
        self.bend_limit_enabled = cd_actuator_dict.get('bendLimitEnabled')
        self.max_range = self.calc_max_range()
        self.q3_norm = self.calc_q3_norm(self.max_range)
        self.q_scaling = 1.0
        self.q3 = 1.0
        self.bending_matrix = self.calc_bending_matrix()
        self.q4_norm = self.calc_q4_norm()
        self.q4 = 1.0

        print('CDActuator Class Constructor')
        print('CD actuator name:', self.name)
        print('CD actuator resolution =', self.resolution)

        # END Constructor
    def calc_max_range(self):
        '''
        calculates the maximum expected range for a cd actuator in the cd actuator beam

        Calling Syntax: max_range = calc_max_range()
        '''
        max_range_down = np.max(np.abs(self.desired_setpoints - self.min_setpoint))
        max_range_up = np.max(np.abs(self.max_setpoint - self.desired_setpoints))
        max_range = np.max([max_range_down, max_range_up])
        return max_range
    
    def calc_q3_norm(self, max_range):
        '''
        Calling syntax: q3_norm = calc_q3_norm(max_range)

        Input Parameters:
        max_range -         the related maximum expected range for cd actuators in the cd actuator beam

        Output parameters:
        q3_norm -           the q3 normalization divisor  
        '''
        q3_norm = max_range**2
        return q3_norm

    def calc_q_scaling(self, R_row_sum_j, Ny):
        '''
        Calculates the q scaling divisor that is required to match the static CD Performance Prediction
        with the steady state of the dynamic CD-MPC close loop solution. See ch 6 in James Fan's Ph.D. thesis 
        for details

        Calling syntax:  q3_scaling = calc_q3_scaling(R_row_sum, Ny)

        Input parameters:
        R_row_sum_j -       The row sum of the ration matrix R for this cd actuator j, see CDMPC object for
                            details.
        Ny -                Number of measurement arrays

        Output parametersç
        q3_scaling -        the q3 scaling divisor
        '''
        q_scaling = R_row_sum_j/Ny
        return q_scaling

    def update_q_scaling(self,  R_row_sum_j, Ny):
         '''
        Updates the attribute q_scaling that is required to match the static CD Performance Prediction
        with the steady state of the dynamic CD-MPC close loop solution. It scales the q3 and q4 weights. 
        See ch 6 in James Fan's Ph.D. thesis for details.

        Calling syntax:  update_q_scaling(R_row_sum, Ny)

        Input parameters:
        R_row_sum_j -       The row sum of the ration matrix R for this cd actuator j, see CDMPC object for
                            details.
        Ny -                Number of measurement arrays

        Output parametersç
        None
        '''
         self.q_scaling = self.calc_q_scaling(R_row_sum_j, Ny)

    def calc_q3(self):
        '''
        Calculates the weighting q3 for devations from ideal setpoints.
        '''
        # Here the energy penalty is the energy penaly provided by the user
        # and the normalization divisor is calculated in the calc_q3_norm method
        # and the scaling divisor is calculated in calc_q3_scaling
        q3 = (self.energy_penalty/self.q3_norm)/self.q_scaling
        return q3
    
    def update_q3(self):
        '''
        updates the q3 attribute
        '''
        self.q3 = self.calc_q3()

    def calc_bending_matrix(self):
        '''
        Calculates the bending matrix
        '''
        nu = self.resolution
        col_1 = np.concatenate((np.array([-2,1]), np.zeros(nu-2)))
        bending_matrix = toeplitz(col_1)
        # Adjust first and last row and use first order difference rather than second order 
        # difference since 2nd order difference connot be defined
        bending_matrix[0][0] = -1
        bending_matrix[nu-1][nu-1] = -1

        return bending_matrix
    
    def calc_q4_norm(self):
        '''
        Calculates the normalization divisor for the q4 (picketing) weighting

        Calling syntax:  q4_norm = calc_q3_norm()

        Input parameters:
        None

        Output parameters:
        q4_norm -       the q4 normalization divisor
        '''
        q4_norm = 1
        nu = self.resolution
        if self.bend_limit_enabled == 1:
            bend_limit = self.bend_limit_second_order
            q4_norm = bend_limit**2
        else:
            max_range = self.calc_max_range()
            q4_norm = max_range**2
        return q4_norm

    def calc_q4(self):
        '''
        Calculates the q4 weight that penalizing cd actuator picketing, bending moments of the cd actuator beam
        '''
        q4 = (self.picketing_penalty/self.q4_norm)/self.q_scaling
        return q4
    
    def update_q4(self):
        '''
        Updates the q4 attribute
        '''
        self.q4 = self.calc_q4()