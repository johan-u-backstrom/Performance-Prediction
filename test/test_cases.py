'''
Module for testing the Performance Prediction Project
'''
import json
import pprint
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import pathlib
import scipy.io as sio
from source import QP
from timeit import default_timer as timer
from source.performance_prediction import CDPerformancePrediction
from source.cd_process_model import CDProcessModel

test_dir =  pathlib.Path(__file__).parent.resolve()
#print('The test directory path is:', test_dir)
project_root_dir = pathlib.Path 
project_root_dir = pathlib.Path(test_dir).parent.resolve()
#print('The proect root directory path is:', project_root_dir)
data_dir = str(project_root_dir) + '/data'
#print('The data directory path is:', data_dir)

def load_performance_prediction_data():
    '''
    Loads the input data for the CDPerformancePrediction Class constructor
    '''
    # Load system data as Dict
    data_file = 'system.json'
    #print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        system_data = json.load(f)
    
    #print('List length:', len(system_data))
    #pprint.pprint(system_data)
   
    # Load cd actuators data as List of Dicts
    data_file = 'cdActuators.json'
    #print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_actuators_data = json.load(f)
    
    #print('List length:', len(cd_actuators_data))
    #print('Keys in first List element:')
    #for key in cd_actuators_data[0]:
    #    print(key)

    # Load cd measurements data as List of Dicts
    data_file = 'cdMeasurements.json'
    #print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_measurements_data = json.load(f)
    
    #print('List length:', len(cd_measurements_data))
    #print('Keys in first List element:')
    #for key in cd_measurements_data[0]:
    #    print(key)

    # Load process model data as Dict
    data_file = 'cdProcessModel.json'
    #print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_process_model_data = json.load(f)
    
    #print('Dict length:', len(cd_process_model_data))
    #pprint.pprint(cd_process_model_data)
    
    # Load CD-MPC tuning data as Dict
    data_file = 'cdMpcTuning.json'
    #print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_mpc_tuning_data = json.load(f)
    
    #print('Dict length:', len(cd_mpc_tuning_data))
    #pprint.pprint(cd_mpc_tuning_data)

    return system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data

def test_case_1(method = 'SLSQP'):
    '''
    test the QP module's solve function
    '''
    PHI = [[2, 0], [0, 8]]
    phi = [-4, -12]
    A = [[1.75, 1],[1, -1]]
    b = [3.5, -1]

    #Starting point, note that this is outside the feasible region
    x0 = np.array([0, 0])

    print('The shape of PHI is:', np.shape(PHI))
    print('The shape of phi is:', np.shape(phi))
    print('The shape of A is:', np.shape(A))
    print('The shape of b is:', np.shape(b))
    print('The shape of x0 is:', np.shape(x0))

    x = QP.solve(PHI, phi, A, b, x0, method)
    print('x =', x)

def test_case_2(method = 'SLSQP'):
    '''
    test the execution of QP matrices exported from
    CD Performance Prediction in Matlab.
    '''
    # load the QP input and output data from the Matlab data file
    data_file = 'QPMatrices.mat'
    data_file_path = data_dir + '/' + data_file
    matlab_data = sio.loadmat(data_file_path, squeeze_me = True)
    print('matlab data variables: ', matlab_data.keys())
    PHI = matlab_data['sPHI_sta']
    phi = matlab_data['sphi_sta']
    A = matlab_data['Ac']
    b = matlab_data['bc']
    dU = matlab_data['dU'] #this is the result of the matlab QP solution

    x0 = np.zeros(np.shape(phi))

    print('The shape of PHI is:', np.shape(PHI))
    print('The shape of phi is:', np.shape(phi))
    print('The shape of A is:', np.shape(A))
    print('The shape of b is:', np.shape(b))
    print('The shape of x0 is:', np.shape(x0))
    print('The shape of dU is:', np.shape(dU))
    
    # get the QP solution 
    start_time = timer()
    x = QP.solve(PHI, phi, A, b, x0, method)
    end_time = timer()
    exec_time = end_time - start_time
    print('x =', x)
    print('Execution time =', exec_time, 'seconds')
    
    # evaluate the solution
    diff = np.round(np.divide(dU - x, dU)*100)
    print('The percent difference between the Matlab and Python QP solution is:', diff)
    diff_avg = np.average(diff)
    print('The average percent difference is:', diff_avg)

    #the quadratic function
    f = lambda x: 0.5*x@PHI@x + phi@x

    f_matlab = f(dU)
    print('The QP function at the Matlab solution = ', f_matlab)
    f_python = f(x)
    print('The QP function at the Python solution = ', f_python)

def test_case_3(data_file = 'cdActuators.json'):
    '''
    Test case 3 tests loading of matlab data that has been saved as a JSON file.
    The test case is made for testing the import of CD Performance Prediction data,
    which is in the form of: array of structs.
    In Python the corresponding form is: list of dictionaries
    The json file has been created using the matlab file: saveDataToJSONFile.m 
    '''
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    
    print('List length:', len(data))
    # pprint.pprint(data[0])
    print('Keys in first List element:')
    for key in data[0]:
        print(key)

def test_case_4():
    '''
    Test case 4 tests the CDPerformancePrediction class constructor.
    
    '''
    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    print('Printing of cd_performance_prediction object attributes')
    print('Nu =', cd_performance_prediction.Nu)
    print('Ny =', cd_performance_prediction.Ny)
    print('G =',  cd_performance_prediction.cd_process_model.G)

    # Print some of the cd_measurement object attributes
    for cd_measurement in cd_performance_prediction.cd_measurements:
        print('CD measurement name:', cd_measurement.name)
        print('CD measurement weight:', cd_measurement.weight)

def test_case_5():
    '''
    Testing of the traditional CD response model matrix (damped cosine). There is a matching test case
    in Matlab that provides the expected result.
    '''
    my = int(100)
    nu = int(30)
    g = 1
    w = 9
    a = 1.2
    d = 0.3
    zba = np.linspace(4.8, 95.2, nu+1)
    response_type = 'even'
   
    G = CDProcessModel.cd_response_matrix_calc(zba, my, nu, g, w, a, d, response_type)
    print('G[:,0] =', G[:,0])
    [fig, ax] = plt.subplots()
    ax.plot(G[:,0])
   
    [fig, ax] = plt.subplots()
    ax.plot(G)

    plt.show()

def test_case_6():
    '''
    Testing of the fiber orientation response model matrix (odd response). There is a matching test case
    in Matlab that provides the expected result.
    '''
    my = 580
    nu = 58
    g = 0.185
    w = 50
    a = 4
    d = 0
    zba = np.linspace(25.6, 558.98, nu+1)
    response_type = 'odd'
    edge_padding_mode = 'reflection'
    G = CDProcessModel.cd_response_matrix_calc(zba, my, nu, g, w, a, d, response_type, edge_padding_mode)
    print('G[:,0] =', G[:,0])
    [fig, ax] = plt.subplots()
    ax.plot(G[:,0])

    [fig, ax] = plt.subplots()
    ax.plot(G[:,int(np.floor(nu/2))])
   
    [fig, ax] = plt.subplots()
    ax.plot(G)

    plt.show()   

def test_case_7():
    '''
    Tests the building of the zone boundary array. There is a matching test case
    in Matlab that provides the expected result.
    '''
    nu = 30
    my = 100
    los = 5.5
    hos = 95.5
    loa = 7
    hoa = 9
    bin_width  = 3 
    awd = 10
    act_widths = awd*np.ones(nu)
    zba = CDProcessModel.zba_calc(los, hos, loa, hoa, bin_width, act_widths)
    
    print('zba =', zba)

def test_case_8():
    '''
    Testing of the actuator edge padding. There is a matching test case
    in Matlab that provides the expected result.
    '''
    my = 580
    nu = 58
    g = 0.185
    w = 50
    a = 4.0
    d = 0
    zba = np.linspace(25.5, 558.98, nu+1)
    response_type = 'damped_sin'
    #response_type = 'damped_cos'
    edge_padding_mode = 'reflection'
    #edge_padding_mode = None
    G = CDProcessModel.cd_response_matrix_calc(zba, my, nu, g, w, a = a, d = d, response_type = response_type, edge_padding_mode = edge_padding_mode)
    print('G[:,0] =', G[:,0])
    
    [fig, ax] = plt.subplots()
    ax.plot(G[:,0])
    plt.title('G[:,0]')

    [fig, ax] = plt.subplots()
    ax.plot(G[:,int(np.floor(nu/2))])
    plt.title('G[:,nu/2]')

    [fig, ax] = plt.subplots()
    ax.plot(G[:,nu-1])
    plt.title('G[:,nu-1]')
   
    [fig, ax] = plt.subplots()
    ax.plot(G)
    plt.title('G')

    plt.show()   

def test_case_9():
    '''
    test the building of the G matrices for the CD mimo system.
    '''
    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Load the G_mimo matrix struct generated from matlab
    data_file = 'G_mimo.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        G_mimo = json.load(f)
    print('rows of G_mimo from matlab:', len(G_mimo))
    print('cols of G_mimo from matlab:', len(G_mimo[0]))
    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    G = cd_performance_prediction.cd_process_model.G
    #G_f_matlab = np.array(G_f_matlab)
    #G_f_diff = G_f_matlab - G_f
    print('Printing of cd_performance_prediction object attributes')
    Nu = cd_performance_prediction.Nu
    Ny = cd_performance_prediction.Ny
    print('Nu =', Nu)
    print('Ny =', Ny)

     # Plot the G matrices from Matlab
    for i in range(Ny):
        for j in range(Nu):
            G_ij = np.array(G_mimo[i][j].get('G'))
            (rows, cols) = G_ij.shape
            print('G_ij min =', np.min(G_ij))
            print('G_ij max = ', np.max(G_ij))
            print('G_ij rows =', rows)
            print('G_ij cols = ', cols)
            [fig, ax] = plt.subplots()
            ax.plot(G_ij)
            title_str = 'G_matlab' + '(' + str(i) + ',' + str(j) + ')'
            plt.title(title_str)

    # Plot the G matrices 
    for i in range(Ny):
        for j in range(Nu):
            G_ij = cd_performance_prediction.cd_process_model.G[i][j]
            (rows, cols) = G_ij.shape
            print('G_ij min =', np.min(G_ij))
            print('G_ij max = ', np.max(G_ij))
            print('G_ij rows =', rows)
            print('G_ij cols = ', cols)
            [fig, ax] = plt.subplots()
            ax.plot(G_ij)
            title_str = 'G' + '(' + str(i) + ',' + str(j) + ')'
            plt.title(title_str)
   
    plt.show()

def test_case_10():
    '''
    test the building of the full G matrix G_f for the CD mimo system.
    This matrix is called ssGainMatrix in the matlab code.
    '''
    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Load the ssGainMatrix generated from matlab
    data_file = 'ssGainMatrix.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        G_f_matlab = json.load(f)

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    G_f = cd_performance_prediction.cd_mpc.G_f
    G_f_matlab = np.array(G_f_matlab)
    G_f_diff = G_f_matlab - G_f
    print('Printing of cd_performance_prediction object attributes')
    print('Nu =', cd_performance_prediction.Nu)
    print('Ny =', cd_performance_prediction.Ny)
    print('G_f =', G_f)
    print('G_f min =', np.min(G_f))
    print('G_f max = ', np.max(G_f))
    
    (rows, cols) = G_f.shape
    print('G_f rows =', rows)
    print('G_f cols = ', cols)

    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, G_f_matlab, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('G_f_matlab')

    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, G_f, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('G_f')

    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, G_f_diff, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('G_f_diff')

    plt.show()

def test_case_11():
    '''
    Tests the building of Q1
    '''
    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Load the Matlab generated Q1 matrix
    data_file = 'Q1.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        Q1_matlab = json.load(f)

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    Y_1 = cd_performance_prediction.cd_mpc.Y_1
    U_1 = cd_performance_prediction.cd_mpc.U_1
    Y_d = cd_performance_prediction.cd_mpc.Y_d
    Q1 = cd_performance_prediction.cd_mpc.Q1
    Q1_matlab = np.array(Q1_matlab)
    Q1_diff = Q1_matlab - Q1

    print('Y(k-1) length:', len(Y_1))
    print('U(k-1) length:', len(U_1))
    print('Y_d(k-1) length:', len(Y_d))
    print('shape of Q1 =', np.shape(Q1))

    for cd_measurement in cd_performance_prediction.cd_measurements:
        print('first valid index for ' + cd_measurement.name + ' is ', cd_measurement.first_valid_index)
        print('last valid index for ' + cd_measurement.name + ' is ', cd_measurement.last_valid_index)
        print('Max expected error for ' + cd_measurement.name + ' is ', cd_measurement.max_expected_error)
        print('q1 normalization for ' + cd_measurement.name + ' is ', cd_measurement.q1_norm)
        print('q1 for ' + cd_measurement.name + ' is ', cd_measurement.q1)


    [fig, ax] = plt.subplots()
    ax.plot(Y_1)
    title_str = 'Y(k-1)' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(U_1)
    title_str = 'U(k-1)' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(Y_d)
    title_str = 'Y_d' 
    plt.title(title_str)

    for cd_measurement in cd_performance_prediction.cd_measurements:
        y_d = cd_measurement.disturbance_profile
        title_str = cd_measurement.name + ' disturbance profile'
        
        [fig, ax] = plt.subplots()
        ax.plot(y_d)
        plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q1))
    title_str = 'Q1_diag' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q1_matlab))
    title_str = 'Q1_diag from Matlab' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q1_diff))
    title_str = 'Q1_diff' 
    plt.title(title_str)

    plt.show()

def test_case_12():
    '''
    Tests the building of the dynamic model
    '''
     # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    Tp = cd_performance_prediction.cd_process_model.time_constant
    Ts = cd_performance_prediction.cd_process_model.sample_time
    num = cd_performance_prediction.cd_process_model.num
    den = cd_performance_prediction.cd_process_model.den

    print('time constant matrix =', Tp)
    print('sample time =', Ts)
    print('numerator matrix =', num)
    print('denominator matrix = ', den)

def test_case_13():
    '''
    testing of method tf2ss in the CDProcessModel Class. A matching 
    test case is available in matlab for comparison. 
    '''
    Tp = 150
    Ts = 20
    G = toeplitz([1, 0.5, 0.1, 0, 0, 0])
    [num, den] = CDProcessModel.num_den_calc(Tp, Ts)
    [A, B, C] = CDProcessModel.tf2ss_calc(num, den, G)  

    print('G =', G)
    print('A =', A)
    print('B =', B)
    print('C =', C)

def test_case_14():
    '''
    testing of method ss_mimo_build in the CDProcessModel Class.
    '''
    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performance_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    # Check the results
    A = cd_performance_prediction.cd_process_model.A
    B = cd_performance_prediction.cd_process_model.B
    C = cd_performance_prediction.cd_process_model.C
    print('Shape of A:', np.shape(A))
    #print('Shape of B:', np.shape(B))
    print('Shape of C:', np.shape(C))
    print('top left 5 x 5 elements of C[2][1]:', C[2][1][0:5][0:5] )

    # Plot A[2][1]
    A_21 = A[2][1]
    (rows, cols) = np.shape(A[2][1])
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, A[2][1], cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('A[2][1]')


     # Plot B[2][1]
    (rows, cols) = np.shape(B[2][1])
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, B[2][1], cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('B[2][1]')

    [fig, ax] = plt.subplots()
    ax.plot(B[2][1])
    plt.title('B[2][1]')

    # Plot C[2][1]
    (rows, cols) = np.shape(C[2][1])
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, C[2][1], cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('C[2][1]')

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(C[2][1]))
    plt.title('Diagonal of C[2][1]')
    
    plt.show()

def test_case_15():
    '''
    Tests the calculation of the R matrix relating the Q3 weighting matrices in 
    CD-MPC steady state prediction and the dynamic CD-MPC controller.
    '''
     # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performance_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    R = cd_performance_prediction.cd_mpc.R
    print('R =', R)

def test_case_16():
    '''
    Tests the calcualtion/building of the Q3 matrix
    '''
     # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Load the Matlab generated Q3 matrix
    data_file = 'Q3.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        Q3_matlab = json.load(f)

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
 
    Q3 = cd_performance_prediction.cd_mpc.Q3
    Q3_matlab = np.array(Q3_matlab)
    Q3_diff = Q3_matlab - Q3

    print('shape of Q3 =', np.shape(Q3))

    for cd_actuator in cd_performance_prediction.cd_actuators:
        print('Energy penalty for ' + cd_actuator.name + ' is ', cd_actuator.energy_penalty) 
        print('Max range for ' + cd_actuator.name + ' is ', cd_actuator.max_range) 
        print('q3 normalization for ' + cd_actuator.name + ' is ', cd_actuator.q3_norm)
        print('q3 for ' + cd_actuator.name + ' is ', cd_actuator.q3)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q3))
    title_str = 'Q3_diag' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q3_matlab))
    title_str = 'Q3_diag from Matlab' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q3_diff))
    title_str = 'Q3_diff' 
    plt.title(title_str)

    plt.show()

def test_case_17():
    '''
    Tests the calcualtion/building of the Q4 matrix
    '''
     # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Load the Matlab generated Q3 matrix
    data_file = 'Q4.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        Q4_matlab = json.load(f)

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    Q4 = cd_performance_prediction.cd_mpc.Q4
    Q4_matlab = np.array(Q4_matlab)
    Q4_diff = Q4_matlab - Q4

    print('shape of Q4 =', np.shape(Q4))

    for cd_actuator in cd_performance_prediction.cd_actuators:
        print('Picketing penalty for ' + cd_actuator.name + ' is ', cd_actuator.picketing_penalty) 
        print('Bending matrix, upper corner for ' + cd_actuator.name + ' is ', cd_actuator.bending_matrix[0:5,0:5]) 
        print('q4 normalization for ' + cd_actuator.name + ' is ', cd_actuator.q4_norm)
        print('q4 for ' + cd_actuator.name + ' is ', cd_actuator.q4)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4))
    title_str = 'Q4_diag' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4_matlab))
    title_str = 'Q4_diag from Matlab' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4_diff))
    title_str = 'Q4_diff' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4, -1))
    title_str = 'Q4 lower off-diag' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4_matlab, -1))
    title_str = 'Q4 lower off-diag from Matlab' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4_diff, -1))
    title_str = 'Q4 lower off-diag diff' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4, 1))
    title_str = 'Q4 upper off-diag' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4_matlab, 1))
    title_str = 'Q4 upper off-diag from Matlab' 
    plt.title(title_str)

    [fig, ax] = plt.subplots()
    ax.plot(np.diag(Q4_diff, 1))
    title_str = 'Q4 upper off-diag diff' 
    plt.title(title_str)

    plt.show()

def test_case_18():
    '''
    Tests the building of the PHI matrix (Hessinan)
    '''
    # load the PHI from the Matlab data file
    data_file = 'QPMatrices.mat'
    data_file_path = data_dir + '/' + data_file
    matlab_data = sio.loadmat(data_file_path, squeeze_me = True)
    print('matlab data variables: ', matlab_data.keys())
    PHI_matlab = matlab_data['sPHI_sta']

    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    PHI = cd_performance_prediction.cd_mpc.PHI

    PHI_diff = PHI_matlab - PHI

    PHI_diff_sum = np.sum(PHI_diff)

    print('sum of differences in PHI =', PHI_diff_sum)
    
    (rows, cols) = np.shape(PHI)
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, PHI, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('PHI')

    (rows, cols) = np.shape(PHI_matlab)
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, PHI_matlab, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('PHI_matlab')

    (rows, cols) = np.shape(PHI_diff)
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, PHI_diff, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('PHI_diff')

    plt.show()

def test_case_19():
    '''
    Test the calculation of the QP matrix phi
    '''
     # load the PHI from the Matlab data file
    data_file = 'QPMatrices.mat'
    data_file_path = data_dir + '/' + data_file
    matlab_data = sio.loadmat(data_file_path, squeeze_me = True)
    print('matlab data variables: ', matlab_data.keys())
    phi_matlab = matlab_data['sphi_sta']

    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    phi = cd_performance_prediction.cd_mpc.phi

    phi_diff = phi_matlab - phi

    phi_diff_sum = np.sum(phi_diff)

    print('sum of differences in phi =', phi_diff_sum)
    
    [fig, ax] = plt.subplots()
    x = range(0,len(phi))
    ax.plot(x, phi, 'b-', label = 'phi')
    ax.plot( x, phi_matlab, 'r--', label = 'phi from matlab')
    ax.legend()
    ax.set_title( 'phi vs phi from matlab')

    plt.show()

def test_case_20():
    '''
    Tests the building of the constraint matrices
    Ac and bc
    Ac@dU(k) <= bc(k)
    where bc(k) = Bc - Cc@U(k-1)

    '''
    # Load the Matlab generated Ac and bc matrices
    data_file = 'Ac.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        Ac_matlab = json.load(f)
    Ac_matlab = np.array(Ac_matlab)

    data_file = 'bc.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        bc_matlab = json.load(f)

    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    Ac = cd_performance_prediction.cd_mpc.Ac
    bc = cd_performance_prediction.cd_mpc.bc
    (n_rows, n_cols) = Ac.shape
    Ac_diff = Ac_matlab - Ac

    print('Top left corner of Ac:', Ac[0:5, 0:5])
    print('Bottom right corner of Ac:', Ac[n_rows-5:n_rows, n_cols-5:n_cols])

    (rows, cols) = np.shape(Ac)
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, Ac, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Ac')

    (rows, cols) = np.shape(Ac_matlab)
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, Ac_matlab, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Ac from matlab')

    (rows, cols) = np.shape(Ac_diff)
    [fig, ax] = plt.subplots(subplot_kw={"projection": "3d"})
    [X, Y] = np.meshgrid(np.linspace(0, cols-1, cols), np.linspace(0, rows-1, rows))  
    surf = ax.plot_surface(X, Y, Ac_diff, cmap = 'coolwarm',linewidth = 0, antialiased = False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Ac diff')

    [fig, ax] = plt.subplots()
    x = range(0,len(bc))
    ax.plot(x, bc, 'b-', label = 'bc')
    ax.plot(x, bc_matlab, 'r-.', label = 'bc from matlab')
    ax.legend()
    ax.set_title( 'bc')

    plt.show()
    
def test_case_21():
    '''
    Tests the calculation of optimal delta u by the QP solver
    '''
    # load the QP input and output data from the Matlab data file
    data_file = 'QPMatrices.mat'
    data_file_path = data_dir + '/' + data_file
    matlab_data = sio.loadmat(data_file_path, squeeze_me = True)
    print('matlab data variables: ', matlab_data.keys())
    dU_matlab = matlab_data['dU'] #this is the result of the matlab QP solution

    # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    dU = cd_performance_prediction.cd_mpc.dU

    [fig, ax] = plt.subplots()
    x = range(0,len(dU))
    ax.plot(x, dU, 'b-', label = 'dU')
    ax.plot(x, dU_matlab, 'r.-', label = 'dU from matlab')
    ax.legend()
    ax.set_title('dU vs dU from Matlab')
    
    for cd_actuator in cd_performance_prediction.cd_actuators:
        du = cd_actuator.du
        [fig, ax] = plt.subplots()
        x = range(0,len(du))
        ax.plot(x, du, 'b-', label = 'optimal du')
        ax.legend()
        ax.set_title(cd_actuator.name)
    plt.show()

def test_case_22():
    '''
    Tests the calculation of the optimal actuator setpoint arrays u(k)
    '''
    # Load the Matlab generated CDActuators struct
    data_file = 'cdActuators.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        cd_actuators_matlab = json.load(f)

     # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    k = 0
    for cd_actuator in cd_performance_prediction.cd_actuators:
        u = cd_actuator.u
        u_matlab = cd_actuators_matlab[k].get('finalProfile')
        k += 1
        [fig, ax] = plt.subplots()
        x = range(0,len(u))
        ax.plot(x, u, 'b-', label = 'optimal u')
        ax.plot(x, u_matlab, 'r-.', label = 'optimal u from matlab')
        ax.legend()
        ax.set_title(cd_actuator.name)
    plt.show()

def test_case_23():
    '''
    Tests the calculation of y(k) in the CDMeasurement objects  
    '''
     # Load the Matlab generated CDMeasurements struct
    data_file = 'cdMeasurements.json'
    data_file_path = data_dir + '/' + data_file
    with open(data_file_path, 'r') as f:
        cd_measurements_matlab = json.load(f)

     # Load the input data for the CDPerformancePrediction Class
    [system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data] = load_performance_prediction_data()

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data, cd_mpc_tuning_data)
    
    k = 0
    for cd_measurement in cd_performance_prediction.cd_measurements:
        y = cd_measurement.y
        two_sigma = round(2*np.std(y)*100)/100
        y_matlab = cd_measurements_matlab[k].get('finalProfile')
        two_sigma_matlab = round(2*np.std(y_matlab)*100)/100
        k += 1
        [fig, ax] = plt.subplots()
        x = range(0,len(y))
        ax.plot(x, y, 'b-', label = 'optimal y')
        ax.plot(x, y_matlab, 'r-.', label = 'optimal y from matlab')
        ax.legend()
        ax.set_title(cd_measurement.name)
        x_label_string = '$2\sigma = $' + str(two_sigma) + ', ' + '$2\sigma_{matlab} = $' + str(two_sigma_matlab)
        ax.set_xlabel(x_label_string)
    plt.show()