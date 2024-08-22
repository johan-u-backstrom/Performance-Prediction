'''
The cd_actuator module implements the CDActuator class
which holds all the cd measurement data and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''


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

        print('CDActuator Class Constructor')
        print('CD actuator name:', self.name)
        print('CD actuator resolution =', self.resolution)