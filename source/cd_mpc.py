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
    Y_1                     The concatenated array of initial profiles, i.e. Y(k-1)
    Q1 -                    The final full (concatinated) Q1 matrix used in the QP optimization
    Q1_norm -               This is the normalization divisor used to calculate Q1, i.e Q1 = Q1_user/Q1_norm 
    Class Methods:
    build_G_full -          Builds the G_f matrix
                                              
    '''

    def __init__(self, cd_mpc_tuning, cd_system, cd_actuators, cd_measurements, cd_process_model):
        '''
        The Class Constructor
        '''
        self.G_f = self.build_G_full(cd_process_model, cd_system, cd_actuators)
        self.Y_1 = self.build_Y_1(cd_measurements)
        self.Q1_norm = self.calc_Q1_norm(cd_measurements)

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
        Builds the concatenated initial Y array, Y(k-1) or Y_1
        '''
        # initialize Y_1
        Ny = len(cd_measurements)
        mY = 0
        for i in range(Ny):
            mY += cd_measurements[i].resolution
        Y_1 = np.zeros(mY)

        # build Y_1
        for i in range(Ny):
            my = cd_measurements[i].resolution
            Y_1[i*my:(i+1)*my] = cd_measurements[i].initial_profile
        
        return Y_1
    
    def calc_Q1_norm(self, cd_measurements):
        '''
        Calculates the Q1_norm divisors used for normalizing Q1 provided by the 
        user: Q1_user.
        '''
        Ny = len(cd_measurements)
        Q1_norm = np.array(np.ones(Ny))
       # for i in range(Ny):
       #     Q1_norm[i] = cd_measurements[i]
           