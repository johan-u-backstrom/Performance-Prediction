'''
The cd_measurement module implements the cdMeasurement class
which holds all the cd measurement data and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''


class CDMeasurement:
    '''
    The CDMeasurement class holds all the cd measurement data and 
    the methods required to process measurement data.

    Calling Syntax:

    Input Parameters:
    cd_measurement - This is a dictionary containing the following cd measurement data

    '''

    def __init__(self, cd_measurement):
        '''
        The Class Constructor
        '''
        self.name =  cd_measurement.get('name')
        self.resolution = cd_measurement.get('resolution')
        self.units = cd_measurement.get('units')
        self.control_mode = cd_measurement.get('controlMode')
        self.initial_profile = cd_measurement.get('initialProfile')
        self.initial_profile_spectrum = cd_measurement.get('initialProfileSpectrum')
        self.final_profile = cd_measurement.get('finalProfile')
        self.final_profile_spectrum = cd_measurement.get('finalProfileSpectrum')
        self.low_edge_of_sheet = cd_measurement.get('lowEdgeOfSheet')
        self.high_edge_of_sheet = cd_measurement.get('highEdgeOfSheet')
        self.md_target = cd_measurement.get('mdTarget')
        self.bias_target = cd_measurement.get('biasTarget')
        self.weight = cd_measurement.get('weight')
        self.weight_norm_factor = cd_measurement.get('measNormFactor')