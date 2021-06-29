import numpy as np
from scipy import optimize
import time

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

def rosen(x):
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

#bounds = [(-5120, 5120), (-5120, 5120)]
bounds = [(-5, 5), (-5, 5)]

results = dict()
start = time.time()
results['shgo'] = optimize.differential_evolution(rosen, bounds)
print(time.time()-start)
print(results['shgo'])