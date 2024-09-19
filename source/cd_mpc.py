'''
The cd_mpc module implements the CDMPC class
which holds all the CD-MPC attributes and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np

class CDMPC:
    '''
    The CDMPC class implements the CD-MPC controller object. At this point the CD-MPC controller is 
    only implemented to solve the steady state problem, which is used in CD-MPC performance predictions.
 
    Calling Syntax:     CDMPC(cd_mpc_tuning, cd_system, cd_actuators, cd_measurements, cd_process_model)

    Input Parameters:
    cd_mpc_tuning -         A cd_mpc_tuning Dictinoary provided by the external caller to the the RESTFul API
    cd_system -             A cd_system object
    cd_actuators -          A List of cd_actuator objects
    cd_measurements -       A List of cd_measurement objects
    cd_process_model -      A cd_process_model object

    Class Attributes:
    G_f -                   The full (concatinated) G matrix for the mimo CD process
    Y_1                     The concatenated array of initial measurement profiles, i.e. Y(k-1)
    U_1                     The concatenated array of initial actuator profiles, i.e. U(k-1)
    Y_d                     The concatenated distrubance array Y_d
    Q1 -                    The final full (concatinated) Q1 matrix used in the QP optimization
   
    Class Methods:
    build_G_full -          Builds the G_f matrix
    build_Y_1               Builds the Y(k-1) array
    build_U_1               Builds the U(k-1) array  
    calc_Y_d                Calculates the Y_d array   
    update_y_d              Updates the disturbance_profile in the CDMeasurement objects
    update_max_exp_e        Updates the max_expected_error in the CDMeasurement objects
    update_q1_norm          Updates the q1 normalization divisor in the CDMeasurement objects
    update_q1               Updates the q1 measurement weight in the CDMeasurement objects

    '''

    def __init__(self, cd_mpc_tuning, cd_system, cd_actuators, cd_measurements, cd_process_model):
        '''
        The Class Constructor
        '''
        self.G_f = self.build_G_full(cd_process_model, cd_system, cd_actuators)
        self.Y_1 = self.build_Y_1(cd_measurements)
        self.U_1 =  self.build_U_1(cd_actuators)
        self.Y_d = self.calc_Y_d(self.Y_1, self.U_1, self.G_f)
        self.update_y_d(self.Y_d, cd_measurements)  # Updates the CDMeasurement objects 
        self.update_max_exp_e(cd_measurements)      # Updates the CDMeasurement objects 
        self.update_q1_norm(cd_measurements)        # Updates the CDMeasurement objects
        self.update_q1(cd_measurements)             # Updates the CDMeasurement objects
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