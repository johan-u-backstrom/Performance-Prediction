'''
The QP module implements a Quadratic Programming (QP) solver used for solving 
QP optimization problems. 
'''
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
#from scipy.optimize import SR1

def solve(PHI, phi, A, b, x0, method):
    '''
    solve solves the QP problem: 

        min f(x) = x'*PHI*x + phi'*x
                                    
    subject to:
        
        A*x <= b

    Inputs:
    PHI -   QP matrix, which is actually the Hessian
    phi -   QP vector
    A -     Linear constraint matrix
    b -     Linear constraint vector

    Outputs:
    The optimal solution x that minimizes the QP problem
    '''
    # the quadratic function
    f = lambda x: 0.5*x@PHI@x + phi@x
    
    # the Hessian
    H = lambda x: PHI

    # the Jacobian
    J = lambda x: PHI@x + phi

    if (method == 'trust-constr'): 
        '''
        Linear inequality constraints for the trust-constr method
        lb <= Ax <= ub
        '''
        linear_constraints = LinearConstraint(A, -np.inf, b)
        '''
        solve the QP probem using the trust-constr method
        '''
        res = minimize(f, x0, method = 'trust-constr', jac = J, hess = H, constraints = linear_constraints, options = {'verbose': 1})
    
    elif (method ==  'SLSQP'):
        '''
        Linear inequality constraints for the SLSQP method
        c(x) = b - Ax >= 0
        '''
        c = lambda x: b - A@x
        linear_constraints = {'type': 'ineq', 'fun': c}
        '''
        solve the QP probem using the SLSQP method, note that the inequality constraints are define differently from the trust-constr method
        '''
        res = minimize(f, x0, method = 'SLSQP', jac = J, constraints = linear_constraints, options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True})

    return res.x

