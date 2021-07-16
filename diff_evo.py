import numpy as np
import time

def easom(x):
    return -np.cos(x[0])*np.cos(x[1])*np.exp(-1*((x[0]-np.pi)**2 + (x[1]-np.pi)**2)) #EASOM
    #return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47))))-x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))#EGGHOLDER
    #return (x[0]**2 + x[1] -11)**2 + (x[0]+x[1]**2 -7)**2   # HIMMEL
    #return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 #ROSEN
    #return 100*np.sqrt(np.abs(x[1]-0.01*x[0]**2)) + 0.01*np.abs(x[0]+10) #Airplane
    #return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2 #Thre hump camel
    #squares = [i**2 - 10*np.cos(2*np.pi*i) for i in x]
    #return 10*len(x) + np.sum(squares)

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])
 
 
# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [np.clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound
 
 
# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = np.random.rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial
 
 
def differential_evolution(pop_size, bounds, iter, F, cr):
    # initialise population of candidate solutions randomly within the specified bounds
    start = time.time()
    pop = bounds[:, 0] + (np.random.rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    #print(pop)
    # evaluate initial population of candidate solutions
    obj_all = [easom(ind) for ind in pop]
    print(time.time()-start)
    # find the best performing vector of initial population
    best_vector = pop[np.argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # run iterations of the algorithm
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = easom(pop[j])
            # compute objective function value for trial vector
            obj_trial = easom(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        pop_std = np.std(obj_all)
        pop_energy = np.abs(np.mean(obj_all))
        #print(pop_std<1e-8*pop_energy)
        print(pop_energy-abs(best_obj))
        if best_obj < prev_obj:
            best_vector = pop[np.argmin(obj_all)]
            prev_obj = best_obj
            # report progress at each iteration
            #print('Iteration: %d f([%s]) = %.16f' % (i, np.around(best_vector, decimals=4), best_obj))
            if abs(pop_energy-abs(best_obj)) < 1e-8 and i>100:
                print('Iteration: %d f([%s]) = %.16f' % (i, np.around(best_vector, decimals=4), best_obj))
                break
        #else:
         #   print('Iteration: %d f([%s]) = %.16f' % (i, np.around(best_vector, decimals=4), best_obj))
    return [best_vector, best_obj]
 
 
# define population size
pop_size = 20*2
# define lower and upper bounds for every dimension
bounds = np.asarray([(-100, 100)]*2)
# define number of iterations
iter = 1000
# define scale factor for mutation
F = 0.8
# define crossover rate for recombination
cr = 0.9
 
# perform differential evolution
start = time.time()
solution = differential_evolution(pop_size, bounds, iter, F, cr)
end = time.time()
print('\nSolution: f([%s]) = %.5f' % (np.around(solution[0], decimals=5), solution[1]))
print('Took {} seconds', end-start)