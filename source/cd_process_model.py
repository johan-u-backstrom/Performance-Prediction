'''
The cd_process_model module implements the cdProcessModel class
which holds all the cd process model attributes and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np

class CDProcessModel:
    '''
    The cdProcessModel class impolements the CD Process Model Object which represents 
    a MIMO (multiple inputs multiple outputs) CD Process. CD Process Model attributes 
    are typically in the form of Ny x Nu matrices, where Ny is the number of measurement arrays
    and Nu is the the number of CD actuator beams. Each element in a Ny x Nu matrice corresponds to a 
    model attribute for a particular CD Actuator beam - CD Measurement array pair. 

    Calling Syntax:

    Input Parameters:
    cd_process_model_dict -     A dictionary containing the following cd process model attributes
    cd_system_obj -             A CDSystem object, see the CDSystem class
    cd_actuators_obj_lst -      A List of CDActuator objects, see the CDActuator class
    cd_measurements_obj_lst -   A List of CDMeasurement objects, see the CDMeasurement class
    Nu -                        Number of CD actuator beams
    Ny -                        Number of CD measurement arrays

    Class Attributes:
    Nu -                        Number of CD actuator beams
    Ny -                        Number of CD measurement arrays
    gain -                      A Ny x Nu matrix (nested list) of CD process model gains
    width -                     A Ny x Nu matrix (nested list) of CD process model widths
    attenuation -               A Ny x Nu matrix (nested list) of CD process model attenuations
    divergence -                A Ny x Nu matrix (nested list) of CD process model divergences
    response_type -             A Ny x Nu matrix (nested list) of CD process model types (functions), 
                                'damped_cos', 'damped_sin', 'inv_prop_decay'. The damped_cos type is the most common and is used for all 
                                CD Processes except for fiber orientation, which uses one of the other two models. 
    edge_padding_mode           A Ny x Nu matrix (nested list) of CD actuator beam (array) padding modes,
                                can be None, 'average', 'linear', or 'reflection'. The default is None. Edge
                                padding is typically only used for fiber orientation process models.
    zba -                       A Ny x Nu matrix (nested list) of Zone Boundary Arrays.
    G -                         A Ny x Nu matrix (nested list) of CD Process Model matrices (nupy 2D arrays), each G[i][j]
                                matrice is of dimension my x nuj, where my is the number of CD bins and 
                                nui is the number of CD actuator zones in CD actuator beam j.
    
    Class Methods:
    response_type_mimo_build    -   builds the response_type attribute
    edge_padding_mode_mimo_build -  builds the edge_padding_mode attribute
    zba_mimo_build -                builds the zba attribute
    G_mimo_build -                  builds the G attribute
    zba_calc -                      calculates a zba[i][j] array
    cd_response_matrix_calc -       calculates a G[i][j] matrix 


                                
    '''

    def __init__(self, cd_process_model_dict, cd_system_obj, cd_actuators_obj_lst, 
                 cd_measurements_obj_lst, Nu, Ny):
        '''
        The Class Constructor
        '''
        self.Ny = Ny
        self.Nu = Nu
        self.gain =  cd_process_model_dict.get('gain')
        self.width = cd_process_model_dict.get('width')
        self.width_in_bins = np.array(np.zeros((Ny, Nu))).tolist()
        # need to convert the width from eng. units to cd bins
        for i in range(Ny):
            for j in range(Nu):
                self.width_in_bins[i][j] = self.width[i][j]/cd_system_obj.bin_width
        self.attenuation = cd_process_model_dict.get('attenuation')
        self.divergence = cd_process_model_dict.get('divergence')
        
        self.response_type = self.response_type_mimo_build(cd_process_model_dict)
        self.edge_padding_mode = self.edge_padding_mode_mimo_build(cd_process_model_dict)
        self.zba = self.zba_mimo_build(cd_system_obj, cd_actuators_obj_lst, cd_measurements_obj_lst, Nu, Ny)
        self.G = self.G_mimo_build(cd_actuators_obj_lst, cd_measurements_obj_lst)
        
        print('CDProcessModel Class Constructor')
        print('process model gain =', self.gain)
        print('process model width in eng units =', self.width)
        print('process model width in bins =', self.width_in_bins)
      


    def response_type_mimo_build(self, cd_process_model_dict):
        '''
        response_type_mimo_build builds/extracts a Ny x Nu 2D nested list of CD response types from the 
        cd_process_model_dict provided by the caller, e.g. RESTFul interface. 
        '''
        Ny = self.Ny
        Nu = self.Nu
        resp_shape = cd_process_model_dict.get('respShape')
        response_type_mimo = np.zeros((Ny, Nu)).tolist()
        for i in range(Ny):
            for j in range(Nu):
                if resp_shape[i][j] == 0:
                    response_type_mimo[i][j] = 'damped_cos'
                elif resp_shape[i][j] == 1:
                    response_type_mimo[i][j] = 'damped_sin'
                elif resp_shape[i][j] == 2:
                    response_type_mimo[i][j] = 'inv_prop_decay'

        return response_type_mimo

    def edge_padding_mode_mimo_build(self, cd_process_model_dict):
        '''
        edge_padding_mimo_build builds/extracts a Ny x Nu 2D nested list of CD edge padding modes from the 
        cd_process_model_dict provided by the caller, e.g. RESTFul interface.
        '''
        Ny = self.Ny
        Nu = self.Nu
        padding_mode = cd_process_model_dict.get('actPaddingMode')
        edge_padding_mode_mimo = np.zeros((Ny, Nu)).tolist()
        for i in range(Ny):
            for j in range(Nu):
                if padding_mode[i][j] == 1:
                    edge_padding_mode_mimo[i][j] = 'average'
                elif padding_mode[i][j] == 2:
                    edge_padding_mode_mimo[i][j] = 'linear'
                elif padding_mode[i][j] == 3:
                    edge_padding_mode_mimo[i][j] = 'reflection'
                else:
                    edge_padding_mode_mimo[i][j] = None
        return  edge_padding_mode_mimo

    def zba_mimo_build(self, cd_system_obj, cd_actuators_obj_lst, cd_measurements_obj_lst, Nu, Ny):
        '''
        zba_mimo_build builds the zba matrix for the Ny x Nu mimo system. It returns a nested 2D List (matrix) of zba arrays.
        '''
        zba_mimo = np.zeros((Ny, Nu)).tolist()
        bin_width = cd_system_obj.bin_width
        for i in range(Ny):
            los = cd_measurements_obj_lst[i].low_edge_of_sheet
            hos = cd_measurements_obj_lst[i].high_edge_of_sheet
            for j in range(Nu):
                loa = cd_actuators_obj_lst[j].low_offset
                hoa = cd_actuators_obj_lst[j].high_offset
                act_width_array = cd_actuators_obj_lst[j].width_array
                zba = self.zba_calc(los, hos, loa, hoa, bin_width, act_width_array)
                zba_mimo[i][j] = zba
        return zba_mimo
    
    def G_mimo_build(self, cd_actuators_obj_lst, cd_measurements_obj_lst):
        '''
        G_mimo_build builds the gain matrices Gs for the Ny x Nu mimo system. It returns a nested 2D List (matrix) of G matrices.
        '''
        Ny = self.Ny
        Nu = self.Nu
        g = self.gain
        w = self.width_in_bins
        a = self.attenuation
        d = self.divergence
        zba = self.zba
        response_type = self.response_type
        edge_padding_mode = self.edge_padding_mode
        G_mimo = np.zeros((Ny, Nu)).tolist()
        for i in range(Ny):
            my = cd_measurements_obj_lst[i].resolution
            for j in range(Nu):
                nu = cd_actuators_obj_lst[j].resolution
                print('Building G_mimo for actuator', cd_actuators_obj_lst[j].name, 'and measurement', cd_measurements_obj_lst[i].name)
                G_mimo[i][j] = CDProcessModel.cd_response_matrix_calc(zba[i][j], my, nu, g[i][j], w[i][j], a[i][j], d[i][j], 
                                                                 response_type[i][j], edge_padding_mode[i][j])
        return G_mimo

    @staticmethod
    def zba_calc(los, hos, loa, hoa, bin_width, act_width_array):
        '''
        zba calculates the Zone Boundary Array for a CD Actuator - CD Measurement pair

        calling syntax: zba = zba(los, hos, loa, hoa, bin_width, act_width_array)
 
        Inputs:
        los -               low edge of sheet, [bins]
        hos -               high edge of sheet, [bins]
        loa -               low actuator offset, [eng. unints]
        hoa -               high actuator offset, [eng. units]
        bin_width -         bin width, [eng.units]
        act_width_array -   actuator width array, [eng. units]

        Outputs:
        zba -                   zone boudary array, [bins]
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
    
    @staticmethod
    def cd_response_matrix_calc(zba, my, nu, g, w, a = 0.0, d = 0.0, response_type = 'damped_cos', edge_padding_mode = None, caller = None):
        '''
        Calculates the cd (spatial) response matrix G for a CD actuator beam - CD measurement array pair.
        The dimension of G is my x nu, where my is the number of CD bins and nu is the number of 
        actuator zones.

        If response_type is 'even', then Dimitri Gorinevsky's spatial model is used, which 
        works for all CD models except for fiber orinetaiton.

        If response_type is 'odd', then Danlei Chu's sptaial model is used, which models the response
        from a slice lip to fiber orinetation.

        The default edge padding mode is none, others are average, linear, and reflection. For details, please see the 
        Experion MX CD Controls User Manual. 

        Calling Syntax:
        G = cd_response_matrix_calc(zba, my, nu, g, w, a = None, d = None, response_type = 'even', edge_padding_mode = None)

        Inputs:
        zba -           the zone boundary array
        my -            number of cd bins       
        nu -            number of cd actuator zones
        g -             model gain
        w -             model width in cd bins
        a -             model attenuation (optional, only used in the even model, defaults to 0)
        d -             model divergence (optional, only used in the even model, defaults to 0)
        response_type   can be 'damped_cos', 'damped_sin', 'inv_prop_decay'. The damped_cos type is the most common and is used for all 
                        CD Processes except for fiber orientation, which uses one of the other two models. 
        edge_padding    Optional, can be  None, 'average', 'linear', or 'reflection', defaults to None
        caller -        Can be 'edge_padding', default is None. To handle the strange Fiber Orientation logic in the matlab code.
       

        Outputs:
        G -         spatial resonse matrix
        '''
        G = np.zeros((my, nu))
        eps = np.finfo(float).eps

        # For fiber orientation models, the original matlab code
        # always uses the inv_prop_decay model for the sheet and only
        # uses the user selected response_type for the edge padding. 
        if caller == None and response_type == 'damped_sin':
            # External call, i.e. main call to the function and damped_sin model
            response_type_used = 'inv_prop_decay'  # force the inv_prop_decay model for the sheet
        else:
            # It is an external call with response_type != 'damped_sin' or a 
            # recursive call from the edge padding logic below
            # -> the user selected response_type is used
            response_type_used = response_type
          
         
        # zba midpoints
        zba_c = np.zeros(nu)
        for i in range(nu):
            zba_c[i] = (zba[i] + zba[i+1])/2
     
        # the cd bin array
        x = np.linspace(1, my, my)
        
        #
        # The response model
        #
        if response_type_used == 'damped_cos':
            # Dimitri's Model
            for i in range(nu):
                G[:,i] = (g/2)*(np.exp(-a*(((x - zba_c[i] - d*w)/w)**2)) * np.cos(np.pi*(x - zba_c[i] - d*w)/w) + 
                                np.exp(-a*(((x - zba_c[i] + d*w)/w)**2)) * np.cos(np.pi*(x - zba_c[i] + d*w)/w))
        elif response_type_used == 'damped_sin':
            # Johan's original Fiber Orientation Model
            for i in range(nu):
                G[:,i] = (g/2)*(np.exp(-a*(((x - zba_c[i] - d*w)/w)**2)) * np.sin(np.pi*(x - zba_c[i] - d*w)/w) + 
                                np.exp(-a*(((x - zba_c[i] + d*w)/w)**2)) * np.sin(np.pi*(x - zba_c[i] + d*w)/w))
        elif response_type_used == 'inv_prop_decay':   
            # Danlei's Fiber Orinetation Model
            for i in range(nu):
                # find the location of CD coordinates where x-ZBAc(i) == 0, to prevent division
                # by zero in the the model
                i_dx_0 = np.argwhere(np.equal(x, zba_c[i]))
                print('i_dx_0 =', i_dx_0)
                if np.size(i_dx_0) != 0: 
                    x[i_dx_0] = eps*np.ones(np.shape(i_dx_0)) + x[i_dx_0]

                impulse_response = (g*w/10.2108)/(x - zba_c[i])*(1 - np.exp(-(16*(x - zba_c[i])/w)**2))
                
                if np.size(i_dx_0) != 0: 
                    impulse_response[i_dx_0] = np.zeros(np.shape(i_dx_0))
                G[:,i] = impulse_response

        #
        # Edge Padding
        #
        if edge_padding_mode != None:
            # Determine the width of spatial response in cd bins and select the widest one
            resp_width = 0      # in cd bins
            for i in range(nu):
                g_max = max(abs(G[:,i]))
                resp_indices = np.argwhere(abs(G[:,i]) > 0.1*g_max)
                if resp_indices.size == 0:
                    break   # We have a process model with no response, e.g. zero gain
                resp_width_i = resp_indices[-1] - resp_indices[0] + 1
                if resp_width_i > resp_width:
                    resp_width = resp_width_i
            print('response width in cd bins =', resp_width)   

            # Determine the number of actuator zones to pad at the low and high end 
            # of the sheet. The required actuator padding is half the response width,
            # measured actuator zones at the measurement location (scanner) 
            awd_at_scanner = np.mean(zba[1:] - zba[0:-1])       # in cd bins
            n_pad = int(np.ceil((resp_width/2)/awd_at_scanner))      # number of actuator zones to pad
            
            print('reqiured act zones to pad = ', n_pad)

            # Augment the zba
            zba_pad_low = np.zeros(n_pad)
            low_pad_start = zba[0] 
            for i in range(-1, -(n_pad+1), -1):
                zba_pad_low[i] = low_pad_start - awd_at_scanner
                low_pad_start = zba_pad_low[i]

            zba_pad_high = np.zeros(n_pad)
            high_pad_start = zba[-1] 
            for i in range(n_pad):
                zba_pad_high[i] = high_pad_start + awd_at_scanner
                high_pad_start = zba_pad_high[i]

            zba_a = np.concatenate((zba_pad_low, zba, zba_pad_high))
            nu_a = int(nu + 2*n_pad)

            print('augmented zba =', zba_a)

            # Calculate the augmented response matrix
            G_aug =  CDProcessModel.cd_response_matrix_calc(zba_a, my, nu_a, g, w, a = a, d = d, response_type = response_type, caller = 'edge_padding')
            print('shape of G_aug:', np.shape(G_aug))
            print('G_aug[:,0,2] = ', G_aug[:,0:3])
            print('G_aug[:,33:35] = ', G_aug[:,33:36])

            # Calculate the equivalent padding matrix G_hat to G_aug, such that
            #
            #   G_hat*u = G_aug*u_aug
            #
            # The effect of padded actuator zones can be represented as changes to the first few and 
            # the last few columns of G.
            G_hat = G
            if edge_padding_mode == 'average':
                # The setpoints of padded zones at the low edge equal the
                # first actuator setpoint.The setpoints of padded zones at
                # the high edge equals the last actuator setpoint:
                #
                #   u_pad_los[i] = u[0] and u_pad_hos[i] = u[-1]
                #
                # This padding model only requires an update to the first and last
                # column of G

                # The required change to the first column of G
                g_hat = G_hat[:,0]                          # modifier for the columns of G
                for i in range(n_pad):
                    g_hat = g_hat + G_aug[:,i]
                G_hat[:,0] = g_hat

                # The required change to the last column of G 
                g_hat = G_hat[:,-1]
                for i in range(n_pad):
                    g_hat = g_hat + G_aug[:, nu_a-n_pad+i]
                G_hat[:,-1] = g_hat
            
            elif edge_padding_mode == 'linear':
                # In this padding mode, the setpoints of padded zones are linear 
                # extensions of the first and last two actuator setpoints:
                #   
                #   u_pad_los[n] = (n+2)*u[0] - (n+1)*u[1] and
                #   u_pad_hos[n] = (n+2)*u[-1] - (n+1)*u[-2], n = 0,1,2,...
                #   
                # where u_pad_los[n] is the nth padded  actuator zone at the low edge
                # and u_pad_hos[n] is the nth padded actuator zone at the high edge.
                # This padding mode will affect the first two and last two columns of G

                # The required change to the first two columns of G
                g0_hat = G_hat[:,0] 
                g1_hat = G_hat[:,1] 
                for i in range(n_pad):
                    g0_hat = g0_hat + (i+2)*G_aug[:,n_pad-1-i]
                    g1_hat = g1_hat - (i+1)*G_aug[:,n_pad-1-i]
                G_hat[:,0] = g0_hat
                G_hat[:,1] = g1_hat

                # The required change to the last two columns of G
                g0_hat = G_hat[:,-1] 
                g1_hat = G_hat[:,-2] 
                for i in range(n_pad):
                    g0_hat = g0_hat + (i+2)*G_aug[:,-n_pad+i]
                    g1_hat = g1_hat - (i+1)*G_aug[:,-n_pad+i]
                G_hat[:,-1] = g0_hat
                G_hat[:,-2] = g1_hat
            
            elif edge_padding_mode == 'reflection':
                # In this padding mode, the setpoints of padded zones are first reflected  
                # in the y-axis and then the x-axis, with origin in [0, u[0]] and [nu-1, u[nu-1]]]
                # for the low and high edge of the sheet respectively:
                #   
                #   u_pad_los[n] = 2*u[0] - u[n] and
                #   u_pad_hos[n] = 2*u[-1] - u[-1-n], n = 0,1,2,...
                #   
                # where u_pad_los[n] is the nth padded  actuator zone at the low edge
                # and u_pad_hos[n] is the nth padded actuator zone at the high edge.
                # This padding mode will affect the first and last n_pad columns of G
                
                # The required change to the first columns of G
                g_hat = G_hat[:,0] 
                for i in range(n_pad):
                    g_hat = g_hat + 2*G_aug[:,i]
                    G_hat[:,i+1] = G_hat[:,i+1] - G_aug[:,n_pad-i-1]
                G_hat[:,0] =g_hat

                # The required change to the last columns of G
                g_hat = G_hat[:,-1] 
                for i in range(n_pad):
                    g_hat = g_hat + 2*G_aug[:,-1-i]
                    G_hat[:,-2-i] = G_hat[:,-2-i] - G_aug[:,-n_pad+i]
                G_hat[:,-1] = g_hat
    
            G = G_hat
   

        # round small leading and trailing values to zero
        epsilon = 0.001*g
        for i in range(nu):
            if abs(G[0][i]) < epsilon:
                first_valid = int(min(np.argwhere(abs(G[:,i]) >  epsilon)))
                G[0:first_valid, i] = np.zeros(first_valid)
    
            if abs(G[my-1][i]) < epsilon:
                last_valid = int(max(np.argwhere(abs(G[:,i]) > epsilon)))
                G[last_valid+1:my, i] = np.zeros(my-1-last_valid)
        
        return G

      