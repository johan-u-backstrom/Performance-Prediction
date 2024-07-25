''' 
This test script test solves a small two dimensional quadratic function with
two linear inequality constraints, i.e. a QP problem, using the minimize function
from scipy with methods trust-constr and SLSQP 
'''
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint

def f(x):
    '''
    inputs:
        x -     numpy array of length = 2
    outputs:
        f(x)    quadratic function with minimum in [x1, x2] = [2, 1.5]
    '''
    f = (x[0] -2)**2 + (2*x[1] -3)**2
    return f

'''
Bounds
inputs to Bounds are [lower_bounds], [upper_bounds]
'''
lower_bounds = [0, 0]
upper_bounds = [5,5]
bounds = Bounds(lower_bounds, upper_bounds)

'''
Linear inequality constraints for the trust-constr method
lb <= Ax <= ub
'''
A = [[1.75, 1],[1, -1]]
lb = [-np.inf, -np.inf]
ub = [3.5, -1]
linear_constraints = LinearConstraint(A, lb, ub)

'''
Linear inequality constraints for the SLSQP method
c(x) = Ax >= 0
'''
def c(x):
    c1 = 3.5 - 1.75*x[0] -x[1]
    c2 = -1 -x[0] + x[1]
    c = np.array([c1, c2])
    return c

'''
Starting point, note that this is outside the feasible region
'''
x0 = np.array([0, 0])

'''
solve the QP probem using the trust-constr method
'''
res = minimize(f, x0, method = 'trust-constr', constraints = [linear_constraints],
               options = {'verbose': 1}, bounds = bounds)

print('----- Results using scipy minimize function with method = trust-constr ----')
print(res.x)
print(res)

'''
solve the QP probem using the SLSQP method, note that both the inequality constraints and
the bounds are define differently.
'''
ineq_const = {'type': 'ineq', 'fun': c}
bounds = ((lower_bounds[0], upper_bounds[0]), (lower_bounds[1], upper_bounds[1]))
res = minimize(f, x0, method = 'SLSQP', constraints = [ineq_const],
               options={'ftol': 1e-9, 'disp': True}, bounds = bounds)

print('----- Results using scipy minimize function with method = SLSQP ----')
print(res.x)
print(res)