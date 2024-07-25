'''
This is the top level run script for running test cases. The top level test run script located in 
the /optimization project root directory is required for maintaing test cases in the /test folder 
and the source code in the /source folder and module importing to work across these folders. 
'''
from test import QP_test_cases

# Methods are: 'trust-constr' or 'SLSQP' 
#QP_test_cases.test_case_1(method = 'trust-constr')
#QP_test_cases.test_case_2(method = 'SLSQP')
QP_test_cases.test_case_2(method = 'trust-constr')

