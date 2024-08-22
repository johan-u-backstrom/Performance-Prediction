'''
The cd_system module implements the CDSystem class
which holds all the System attributes and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''

class CDSystem:
    '''
    The CDSystem class holds all the cd system attributes and methods.


    Calling Syntax:

    Input Parameters:
    cd_system_dict -        This is a dictionary containing the following cd process model attributes

    '''

    def __init__(self, cd_system_dict):
        '''
        The Class Constructor
        '''
        self.number_of_cd_bins = cd_system_dict.get("numberOfCDBins")
        self.bin_width = cd_system_dict.get("binWidth")
        self.spatial_eng_units = cd_system_dict.get("spatialEngineeringUnits")
        self.spatial_disp_units = cd_system_dict.get("SpatialDisplayUnits")

        print('CDSystem Class Constructor')
        print('bin width =', self.bin_width)
        print('number of cd bins =', self.number_of_cd_bins)