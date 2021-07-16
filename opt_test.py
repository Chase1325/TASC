import numpy as np
from scipy import optimize
import time

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

def rosen(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

def himmel(x):
    return (x[0]**2 + x[1] -11)**2 + (x[0]+x[1]**2 -7)**2

def easom(x):
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-1*((x[0]-np.pi)**2 + (x[1]-np.pi)**2))

#bounds = [(-5120, 5120), (-5120, 5120)]
bounds = [(-5, 5), (-5, 5)]

results = dict()
start = time.time()
results['shgo'] = optimize.differential_evolution(easom, bounds, init='latinhypercube', polish=True, disp=True, tol=1e-16)
print(time.time()-start)
print(results['shgo'])