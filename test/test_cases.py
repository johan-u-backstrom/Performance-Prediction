'''
Module for testing the Performance Prediction Project
'''
import json
import pprint
import numpy as np
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
    # Load system data as Dict
    data_file = 'system.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        system_data = json.load(f)
    
    print('List length:', len(system_data))
    pprint.pprint(system_data)
   
    # Load cd actuators data as List of Dicts
    data_file = 'cdActuators.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_actuators_data = json.load(f)
    
    print('List length:', len(cd_actuators_data))
    print('Keys in first List element:')
    for key in cd_actuators_data[0]:
        print(key)

    # Load cd measurements data as List of Dicts
    data_file = 'cdMeasurements.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_measurements_data = json.load(f)
    
    print('List length:', len(cd_measurements_data))
    print('Keys in first List element:')
    for key in cd_measurements_data[0]:
        print(key)

    # Load process model data as Dict
    data_file = 'cdProcessModel.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_process_model_data = json.load(f)
    
    print('Dict length:', len(cd_process_model_data))
    pprint.pprint(cd_process_model_data)

    # Create a cd_performanc_prediction object
    cd_performance_prediction = CDPerformancePrediction(system_data, cd_actuators_data, cd_measurements_data, cd_process_model_data)
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
    my = 128
    nu = 30
    g = 1
    w = 9
    a = 1.2
    d = 0.3
    zba = np.linspace(4.5, 124.5, nu+1)
    response_type = 'odd'
    edge_padding_mode = 'reflection'
   
    G = CDProcessModel.cd_response_matrix_calc(zba, my, nu,g,w,response_type = response_type, edge_padding_mode = edge_padding_mode)
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
    # Load system data as Dict
    data_file = 'system.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        system_data = json.load(f)
    
    print('List length:', len(system_data))
    pprint.pprint(system_data)
   
    # Load cd actuators data as List of Dicts
    data_file = 'cdActuators.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_actuators_data = json.load(f)
    
    print('List length:', len(cd_actuators_data))
    print('Keys in first List element:')
    for key in cd_actuators_data[0]:
        print(key)

    # Load cd measurements data as List of Dicts
    data_file = 'cdMeasurements.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_measurements_data = json.load(f)
    
    print('List length:', len(cd_measurements_data))
    print('Keys in first List element:')
    for key in cd_measurements_data[0]:
        print(key)

    # Load process model data as Dict
    data_file = 'cdProcessModel.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_process_model_data = json.load(f)
    
    print('Dict length:', len(cd_process_model_data))
    pprint.pprint(cd_process_model_data)

    # Load CD-MPC tuning data as Dict
    data_file = 'cdMpcTuning.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_mpc_tuning_data = json.load(f)
    
    print('Dict length:', len(cd_mpc_tuning_data))
    pprint.pprint(cd_mpc_tuning_data)
    
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
    # Load system data as Dict
    data_file = 'system.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        system_data = json.load(f)
    
    print('List length:', len(system_data))
    pprint.pprint(system_data)
   
    # Load cd actuators data as List of Dicts
    data_file = 'cdActuators.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_actuators_data = json.load(f)
    
    print('List length:', len(cd_actuators_data))
    print('Keys in first List element:')
    for key in cd_actuators_data[0]:
        print(key)

    # Load cd measurements data as List of Dicts
    data_file = 'cdMeasurements.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_measurements_data = json.load(f)
    
    print('List length:', len(cd_measurements_data))
    print('Keys in first List element:')
    for key in cd_measurements_data[0]:
        print(key)

    # Load process model data as Dict
    data_file = 'cdProcessModel.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_process_model_data = json.load(f)
    
    print('Dict length:', len(cd_process_model_data))
    pprint.pprint(cd_process_model_data)
    
    # Load CD-MPC tuning data as Dict
    data_file = 'cdMpcTuning.json'
    print('data_file = ', data_file)
    data_file_path = data_dir + '/' + data_file
   
    with open(data_file_path, 'r') as f:
        cd_mpc_tuning_data = json.load(f)
    
    print('Dict length:', len(cd_mpc_tuning_data))
    pprint.pprint(cd_mpc_tuning_data)

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
