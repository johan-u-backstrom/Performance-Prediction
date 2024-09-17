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
  
    Class Methods:
    build_G_full -          Builds the G_f matrix
                              
    '''

    def __init__(self, cd_mpc_tuning, cd_system, cd_actuators, cd_measurements, cd_process_model):
        '''
        The Class Constructor


        '''
        self.G_f = self.build_G_full(cd_process_model, cd_system, cd_actuators)

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
        