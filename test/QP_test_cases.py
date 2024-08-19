'''
Module for testing the  QP module
'''
import json
import pprint
import numpy as np
import pathlib
import scipy.io as sio
from source import QP
from timeit import default_timer as timer


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