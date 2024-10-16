'''
The cd_mpc module implements the CDMPC class
which holds all the CD-MPC attributes and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np
from scipy.linalg import block_diag

class CDMPC:
    '''
    The CDMPC class implements the CD-MPC controller object. At this point the CD-MPC controller is 
    only implemented to solve the steady state problem, which is used in CD-MPC performance predictions.
 
    Calling Syntax:     CDMPC(cd_mpc_tuning, cd_system, cd_actuators, cd_measurements, cd_process_model)

    Input Parameters:
    cd_mpc_tuning -         A cd_mpc_tuning Dictinoary provided by the external caller to the the RESTFul API
    cd_system -             A cd_system object instantiated from the CDSystem class
    cd_actuators -          A List of cd_actuator objects instantiated from the CDActuator class
    cd_measurements -       A List of cd_measurement objects instantiated from the CDMeasurement class
    cd_process_model -      A cd_process_model object instantiated from the CDProcessModel class

    Class Attributes:
    G_f -                   Full (concatinated) G matrix for the mimo CD process
    Y_1                     Concatenated array of initial measurement profiles, i.e. Y(k-1)
    U_1                     Concatenated array of initial actuator profiles, i.e. U(k-1)
    Y_d                     Cconcatenated distrubance array Y_d
    Q1 -                    Final full (concatinated) Q1 matrix used in the CD-MPC objective function
    R -                     Ratio matrix relating Q3 of the steady state preformance prediction problem and the dynamic CD-MPC controller
    R_row_sum               The row sum of the ration matrix R
    Q3 -                    Final full (concatinated) Q3 matrix used in the CD-MPC objective function
   
    Class Methods:
    build_G_full -          Builds the G_f matrix
    build_Y_1               Builds the Y(k-1) array
    build_U_1               Builds the U(k-1) array  
    calc_Y_d                Calculates the Y_d array   
    update_y_d              Updates the disturbance_profile in the CDMeasurement objects
    update_max_exp_e        Updates the max_expected_error in the CDMeasurement objects
    update_q1_norm          Updates the q1 normalization divisor in the CDMeasurement objects
    update_q1               Updates the q1 measurement weight in the CDMeasurement objects
    calc_Q1                 Calculates the Q1 (measurement) weighting matrix
    calc_ratio_matrix       Calculates the ratio matrix R
    calc_R_row_sum          Calculates the row sum of R
    update_q3_norm          Updates the q3 normalization divisor in the CDActuator object
    update_q3_scaling       Updates the q3 scaling divisor in the CDActuator object
    update_q3               Updates the q3 weight (energy weight) in the CDActuator object
    calc_Q3                 Calculates the Q3 (energy) weighting matrix

    '''

    def __init__(self, cd_mpc_tuning, cd_system, cd_actuators, cd_measurements, cd_process_model):
        '''
        The Class Constructor
        '''
        # Class Attributes
        self.Hu = cd_mpc_tuning.get('Hu')
        self.Hp = cd_mpc_tuning.get('Hp')
        self.G_f = self.build_G_full(cd_process_model, cd_system, cd_actuators)
        self.Y_1 = self.build_Y_1(cd_measurements)
        self.U_1 =  self.build_U_1(cd_actuators)
        self.Y_d = self.calc_Y_d(self.Y_1, self.U_1, self.G_f)
        
        # Calculate Q1 
        self.update_y_d(self.Y_d, cd_measurements)  # Updates the CDMeasurement objects 
        self.update_max_exp_e(cd_measurements)      # Updates the CDMeasurement objects 
        self.update_q1_norm(cd_measurements)        # Updates the CDMeasurement objects
        self.update_q1(cd_measurements)             # Updates the CDMeasurement objects
        self.Q1 = self.calc_Q1(cd_measurements)

        # Calculate Q3
        self.R = self.calc_ratio_matrix(cd_process_model)
        self.R_row_sum = self.calc_ratio_matrix_row_sum()
        self.update_q_scaling(cd_actuators, cd_measurements)    # Updates the CDActuator objects       
        self.update_q3(cd_actuators)                            # Updates the CDActuator objects
        self.Q3 = self.calc_Q3(cd_actuators)

        # Calculate Q4
        self.update_q4(cd_actuators)                                        # Updates the CDActuator objects
        self.Q4 = self.calc_Q4(cd_actuators)
    # END Constructor

    def build_G_full(self, cd_process_model, cd_system, cd_actuators):
        '''
        builds the full G matrix G_f for the mimo (multiple cd measurements multiple cd actuators beams) CD process.

                                             Nu
        The dimension of G_f is:  Ny*my x sum(nu(i)) 
                                             i=1
        
        This matrix is used in the steady state performance prediction of a mimo CD process controlled by a CD-MPC
        controller.

        Calling Syntax: G_f = build_G_full(cd_process_model)

        Input Parameters:
        cd_process_model -              A cd_process_model object
        cd_system -                     A cd_system object
        cd_actuators -                  A List of cd_actuator objects

        '''
        # Initialize G_f to an numpy array of zeros
        my = int(cd_system.number_of_cd_bins)
        Ny = int(cd_process_model.Ny)
        Nu = int(cd_process_model.Nu)
        my_f = Ny*my
        nu_f = int(0)
        for j in range(Nu):
            nu_f += int(cd_actuators[j].resolution)   
        G_f = np.zeros((my_f, nu_f))

        # Build G_f, which is the concatenation of the G[i][j] matrices in the CDProcessModel object,
        # i.e. the SISO CD matrices in the multivariable CD process.
        for i in range(Ny):
            nu_sum  = 0
            for j in range(Nu):
                nu_j = cd_actuators[j].resolution
                G_ij = cd_process_model.G[i][j]
                G_f[i*my:(i+1)*my, nu_sum:nu_sum+nu_j] = G_ij
                nu_sum += nu_j
        
        return G_f
        
    def build_Y_1(self, cd_measurements):
        '''
        Builds the concatenated  Y(k-1) array that is used in the QP problem formulation for 
        the CD-MPC controller
        '''
        Y_1 = [] 
        for cd_measurement in cd_measurements:
            Y_1 += cd_measurement.initial_profile
        Y_1 = np.array(Y_1)
        return Y_1
    
    def build_U_1(self, cd_actuators):
        '''
        Builds the concatenated U(k-1) array that is used in the QP problem formulation for 
        the CD-MPC controller
        '''
        U_1 = [] 
        for cd_actuator in cd_actuators:
            U_1 += cd_actuator.initial_profile
        U_1 = np.array(U_1)
        return U_1

    def calc_Y_d(self, Y_1, U_1, G_f):
        '''
        Calculates the concatenated distrurbance profile Y_d 
        '''
        Y_d = Y_1 - G_f@U_1
        return Y_d
    
    def update_y_d(self, Y_d, cd_measurements):
        '''
        Extracts the individual measuremewnt disturbance profiles y_d[i]
        from Y_d and updates the CDMeasurement objects
        '''
        Ny = len(cd_measurements)
        for i in range(Ny):
            my = cd_measurements[i].resolution
            y_d = Y_d[i*my:(i+1)*my]
            cd_measurements[i].set_disturbance_profile(y_d)
    
    def update_max_exp_e(self, cd_measurements):
        '''
        Updates the maximum expected error for the CDMeasurement objects.
        Note that this function should be called after update_y_d has been called.
        '''
        for cd_measurement in cd_measurements:
            cd_measurement.calc_max_expected_error()

    def update_q1_norm(self, cd_measurements):
        '''
        Updates the q1 normalization divisor for the CDMEasurement objects. 
        Note that this function should be called after update_max_exp_e has been called. 
        '''
        for cd_measurement in cd_measurements:
            cd_measurement.calc_q1_norm()
   
    def update_q1(self, cd_measurements):
        '''
        Updates the q1 measurement weight for the CDMeasurement object.
        '''
        for cd_measurement in cd_measurements:
            cd_measurement.calc_q1()
    
    def calc_Q1(self, cd_measurements):
        '''
        Calculates the measurement weighting matrix Q1 that defines the relative
        importanace between the sheet properties and is used on the CD-MPC QP objective
        function.
        '''
        Q1_list = []
        for cd_measurement in cd_measurements:
            Q1_list += [cd_measurement.q1]*cd_measurement.resolution 
        Q1_array = np.array(Q1_list)
        Q1 = np.diag(Q1_array)
        return Q1
      
    def calc_ratio_matrix(self, cd_process_model):
        '''
        calculates the ratio matrix R that is used to adjust the Q3 weights of the 
        CD-MPC steady state performance prediction QP problem such that the results 
        matches the result of the dynamic CD-MPC controller in steady state.

        Q3_ss^-1 = R.*Q3^-1, where .* denotes element wise multiplication (Matlab syntax)

        See chapter 6 in Junquang Fan's PhD thesis for details.

        Calling syntax: R=calc_ratio_matrix(cd_process_model)

        Inputs:
		    cd_process_model -  A CDProcessModel object     

        Outputs:
		    R -                 The Q3 ratio matrix
        '''
        Ny = cd_process_model.Ny
        Nu = cd_process_model.Nu
        Td = cd_process_model.time_delay
        Hp = self.Hp
        A = cd_process_model.A
        B = cd_process_model.B
        
        R = np.zeros((Ny, Nu))

        for i in range(Ny):
            for j in range(Nu):
                if np.mean(np.diag(B[i][j])) != 0:
                    # There is a non zero gain between actuator beam j and measurement array i,
                    # hence we need to update R from its initial values of zero
                    a = np.mean(np.diag(A[i][j]))
                    a_sum = 0
                    for m in range(1,Hp-Td[i][j]+1):
                        for n in range(1, m+1):
                            a_sum += a**(n-1)
                    R[i][j] = (1-a)*a_sum   
        return R
    
    def calc_ratio_matrix_row_sum(self):
        '''
        Calculates the row sum of the ratio matrix R.

        Calling syntax: R_row_sum = calc_ratio_matrix_row_sum()

        Inputs:
		    None    

        Outputs:
		    R_row_sum -         The row sum of the ratio matrix
        '''
        R = self.R
        R_row_sum = np.sum(R, axis = 0)

        # R_row_sum is part of the denominator in the Q3 normalization, so it cannot be zero.
        # A R_row_sum element of zero indicates that a CD actuator beam has no impact on any CD measurement
        # array, so this should never happen but the matlab code has this protection ....
        for j in range(len(R_row_sum)):
            if R_row_sum[j] == 0:
                R_row_sum = 1e-8

        return R_row_sum

    def update_q_scaling(self, cd_actuators, cd_measurments):
        '''
        updates the q scaling for each CDActuator object in the cd_actuators list of CDActuator objects
        '''
        Ny = len(cd_measurments)
        Nu = len(cd_actuators)
        R_row_sum = self.R_row_sum
        for j in range(Nu):
            cd_actuators[j].update_q_scaling(R_row_sum[j], Ny)
    
    def update_q3(self, cd_actuators):
        '''
        updates the q3 weight for each CDActuator object in the cd_actuators list of CDActuator objects 
        '''
        for cd_actuator in cd_actuators:
            cd_actuator.update_q3()
        
    def calc_Q3(self, cd_actuators):
        '''
        Calculates the Q3 weighting matrix for the cd performance prediction
        QP problem. Q3 penalizes deviations from nominal (ideal) CD actuator setpoints.
        We also refer this to an energy penalty since the ideal CD actuator setpoints are
        often the ones that minimize enery usage.
        '''
        Q3_list = []
        for cd_actuator in cd_actuators:
            Q3_list += [cd_actuator.q3]*cd_actuator.resolution
        Q3_array = np.array(Q3_list)
        Q3 = np.diag(Q3_array)
        return Q3
    
    def update_q4(self, cd_actuators):
        '''
        updates the q4 weight for each CDActuator object in the cd_actuators list of CDActuator objects 
        '''
        for cd_actuator in cd_actuators:
            cd_actuator.update_q4()

    def calc_Q4(self, cd_actuators):
        '''
        Calculates the Q4 weighting matrix for the cd performance prediction
        QP problem. Q4 penalizes cd actuator picketing (every other actuator up every other actuator down)
        or bending of the cd actuator beam. It penalizing the 2nd order differnece of the cd actuator array.
        '''
        Q4_list = []
        for cd_actuator in cd_actuators:
            # First, we create a list of the picketing penalty matrices, one for each cd actuator beam
            q4_matrix = np.diag(cd_actuator.q4*np.ones(cd_actuator.resolution))
            bending_matrix = cd_actuator.bending_matrix
            Q4_list.append(np.transpose(bending_matrix)@q4_matrix@bending_matrix)
        print('Q4_list =', Q4_list)
        print('Shape of Q4_list = ', np.shape(Q4_list))
        # Second, we build the block diagonal Q4 matrix
        Q4 = block_diag(*Q4_list)
    
        return Q4
        