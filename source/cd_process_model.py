'''
The cd_process_model module implements the cdProcessModel class
which holds all the cd process model attributes and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np

class CDProcessModel:
    '''
    The cd_process_model module implements the cdProcessModel class
    which holds all the cd process model attributes and methods.

    Calling Syntax:

    Input Parameters:
    cd_process_model_dict - This is a dictionary containing the following cd process model attributes
    Nu -                    Number of CD actuator beams
    Ny -                    Number of CD measurement arrays
    '''

    def __init__(self, cd_process_model_dict, cd_system_obj, cd_actuators_obj_lst, 
                 cd_measurements_obj_lst, Nu, Ny):
        '''
        The Class Constructor
        '''
        self.gain =  cd_process_model_dict.get('gain')
        self.width = cd_process_model_dict.get('width')
        self.attenuation = cd_process_model_dict.get('attenuation')
        self.divergence = cd_process_model_dict.get('divergence')
        self.edge_padding_mode = cd_process_model_dict.get('actPaddingMode')
    
        self.zba_matrix = self.zba_matrix_calc(cd_system_obj, cd_actuators_obj_lst, cd_measurements_obj_lst, Nu, Ny)
        
        print('CDProcessModel Class Constructor')
        print('process model gain =', self.gain)
        print('process model width =', self.width)
        print('num cols of the zba_matrix List:', len(self.zba_matrix))
        print('num rows of the zba_matrix List:', len(self.zba_matrix[0]))
        print('zba_matrix[0][0] =', self.zba_matrix[0][0])
        print('zba_matrix[0][1] =', self.zba_matrix[0][1])
        print('zba_matrix[1][0] =', self.zba_matrix[1][0])
        print('zba_matrix[1][1] =', self.zba_matrix[1][1])
        print('zba_matrix[2][0] =', self.zba_matrix[2][0])
        print('zba_matrix[2][1] =', self.zba_matrix[2][1])


    def zba_matrix_calc(self, cd_system_obj, cd_actuators_obj_lst, cd_measurements_obj_lst, Nu, Ny):
        '''
        zba_matrix_calc generates the zba matrix which is a Ny x Nu
        nested List of zba arrays
        '''
        zba_matrix = np.zeros((Ny, Nu)).tolist()
        bin_width = cd_system_obj.bin_width
        num_bins = cd_system_obj.number_of_cd_bins
        for i in range(Ny):
            los = cd_measurements_obj_lst[i].low_edge_of_sheet
            hos = cd_measurements_obj_lst[i].high_edge_of_sheet
            for j in range(Nu):
                loa = cd_actuators_obj_lst[j].low_offset
                hoa = cd_actuators_obj_lst[j].high_offset
                act_width_array = cd_actuators_obj_lst[j].width_array
                zba = self.zba_calc(los, hos, loa, hoa, bin_width, num_bins, act_width_array)
                zba_matrix[i][j] = zba
        return zba_matrix
    
    def zba_calc(self, los, hos, loa, hoa, bin_width, num_bins, act_width_array):
        '''
        zba calculates the Zone Boundary Array for a CD Actuator - CD Measurement pair

        calling syntax: zba = zba(los, hos, loa, hoa, bin_width, num_bins, act_width_array)
 
        Inputs:
        los -               low edge of sheet, [bins]
        hos -               high edge of sheet, [bins]
        loa -               low actuator offset, [eng. unints]
        hoa -               high actuator offset, [eng. units]
        bin_width -         bin width, [eng.units]
        act_width_array -   actuator width array, [eng. units]

        Outputs:
        zba -                   zone boudary array, [bins]
        zba_c -                 zone boundary array midpoints, [bins]
        '''
        loa = loa/bin_width                                         # in bins
        hoa = hoa/bin_width                                         # in bins
        act_width_array = np.array(act_width_array)/bin_width       # in bins   
        N = len(act_width_array)                                    # number of actuators 
        tot_act_width = np.sum(act_width_array)                     # in bins				            
        zba = np.zeros(N + 1)
        
        # slope = (1-shrinkage)
        slope = (hos-los)/(tot_act_width - loa - hoa)
   
        # intercept = zba(1)
        intercept = los - loa*slope
   
        # calculate the zba
        act_width_array_at_scanner = act_width_array*slope			
        zba[0] = intercept
        for i in range(1, N+1):
            zba[i] = zba[i-1] + act_width_array_at_scanner[i-1]
        
        return zba