'''
The cd_process_model module implements the cdProcessModel class
which holds all the cd process model attributes and methods.

Copyright: Honeywell Process Solutions - North Vancouver
'''
import numpy as np

class CDProcessModel:
    '''
    The cdProcessModel class
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
        for i in range(Ny):
            los = cd_measurements_obj_lst[i].low_edge_of_sheet
            hos = cd_measurements_obj_lst[i].high_edge_of_sheet
            for j in range(Nu):
                loa = cd_actuators_obj_lst[j].low_offset
                hoa = cd_actuators_obj_lst[j].high_offset
                act_width_array = cd_actuators_obj_lst[j].width_array
                zba = self.zba_calc(los, hos, loa, hoa, bin_width, act_width_array)
                zba_matrix[i][j] = zba
        return zba_matrix
    
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
    def cd_response_matrix_build(zba, my, nu, g, w, a = None, d = None, response_type = 'even', edge_padding_mode = None):
        '''
        Builds the cd (spatial) response matrix G for a CD actuator beam - CD measurement array pair.
        The dimension of G is my x nu, where my is the number of CD bins and nu is the number of 
        actuator zones.

        If response_type is 'even', then Dimitri Gorinevsky's spatial model is used, which 
        works for all CD models except for fiber orinetaiton.

        If response_type is 'odd', then Danlei Chu's sptaial model is used, which models the response
        from a slice lip to fiber orinetation.

        The default edge padding mode is none, others are average, linear, and reflection. For details, please see the 
        Experion MX CD Controls User Manual. 

        Calling Syntax:
        G = cd_response_matrix_build(zba, my, nu, g, w, a = None, d = None, response_type = 'even', edge_padding_mode = None)

        Inputs:
        zba -           the zone boundary array
        my -            number of cd bins       
        nu -            number of cd actuator zones
        g -             model gain
        w -             model width
        a -             model attenuation (optional, only used in the even model)
        d -             model divergence (optional, only used in the even model)
        response_type   can be 'even' or 'odd' (optional, defaults to 'even')
        edge_padding    Optional, can be  'average', 'linear', or 'reflection'
       

        Outputs:
        G -         spatial resonse matrix
        '''
        G = np.zeros((my, nu))
        eps = np.finfo(float).eps

        # zba midpoints
        zba_c = np.zeros(nu)
        for i in range(nu):
            zba_c[i] = (zba[i] + zba[i+1])/2
     
        # the cd bin array
        x = np.linspace(1, my, my)
        
        #
        # The response model
        #
        if response_type == 'even':
            # Dimitri's Model
            for i in range(nu):
                G[:,i] = (g/2)*(np.exp(-a*(((x - zba_c[i] - d*w)/w)**2)) * np.cos(np.pi*(x - zba_c[i] - d*w)/w) + 
                                np.exp(-a*(((x - zba_c[i] + d*w)/w)**2)) * np.cos(np.pi*(x - zba_c[i] + d*w)/w))
        elif response_type == 'odd':   
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
            G_aug =  CDProcessModel.cd_response_matrix_build(zba_a, my, nu_a, g, w, a = a, d = d, response_type = response_type)
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

      